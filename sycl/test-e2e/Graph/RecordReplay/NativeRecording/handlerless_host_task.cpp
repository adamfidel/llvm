// REQUIRES: level_zero_v2_adapter && arch-intel_gpu_bmg_g21

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if level_zero %{%{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}

// Test that handlerless host_task (via ext::oneapi::experimental::host_task)
// works in native recording mode. The host task should execute as part of the
// graph, ordered between kernels.

#include "../../graph_common.hpp"

#include <sycl/ext/oneapi/experimental/enqueue_functions.hpp>
#include <sycl/properties/all_properties.hpp>

int main() {
  queue Queue{property::queue::in_order{}};

  exp_ext::command_graph Graph{
      Queue.get_context(),
      Queue.get_device(),
      {exp_ext::property::graph::enable_native_recording{}}};

  constexpr size_t N = 1024;
  uint32_t *Data = malloc_shared<uint32_t>(N, Queue);

  Graph.begin_recording(Queue);

  // Kernel 1: initialize data
  Queue.submit([&](handler &CGH) {
    CGH.parallel_for(range<1>{N}, [=](id<1> idx) { Data[idx] = idx + 1; });
  });

  // Handlerless host_task: multiply each element by 3
  exp_ext::host_task(Queue, [=]() {
    for (size_t i = 0; i < N; i++) {
      Data[i] = Data[i] * 3;
    }
  });

  // Kernel 2: add 10 to each element
  Queue.submit([&](handler &CGH) {
    CGH.parallel_for(range<1>{N},
                     [=](id<1> idx) { Data[idx] = Data[idx] + 10; });
  });

  Graph.end_recording(Queue);

  auto ExecutableGraph = Graph.finalize();
  Queue.submit([&](handler &CGH) { CGH.ext_oneapi_graph(ExecutableGraph); });
  Queue.wait();

  // Verify results:
  // Kernel 1: Data[i] = i + 1
  // Host task: Data[i] = (i + 1) * 3
  // Kernel 2: Data[i] = (i + 1) * 3 + 10
  for (size_t i = 0; i < N; i++) {
    uint32_t Expected = (i + 1) * 3 + 10;
    assert(check_value(i, Expected, Data[i], "Data"));
  }

  free(Data, Queue);
  return 0;
}
