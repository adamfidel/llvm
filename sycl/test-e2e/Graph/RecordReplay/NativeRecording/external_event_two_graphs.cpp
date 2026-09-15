// REQUIRES: level_zero_v2_adapter && arch-intel_gpu_bmg_g21

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if level_zero %{%{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}

// Uses two external event to synchronize back and forth between two graphs

#include "../../graph_common.hpp"

#include <sycl/ext/oneapi/experimental/reusable_events.hpp>
#include <sycl/properties/all_properties.hpp>

int main() {
  device Dev;
  context Ctx{Dev};

  queue Queue1{Ctx, Dev, {property::queue::in_order{}}};
  queue Queue2{Ctx, Dev, {property::queue::in_order{}}};

  QueueStateVerifier verifier(Queue1, Queue2);

  exp_ext::properties External{exp_ext::graph_external{}};
  // Fresh events are already complete, so the first replay of Graph1 does not
  // block on EventA.
  event EventA = exp_ext::make_event(Ctx);
  event EventB = exp_ext::make_event(Ctx);

  const size_t N = 1024;
  int *Data = malloc_device<int>(N, Dev, Ctx);
  int *Staging = malloc_device<int>(N, Dev, Ctx);

  std::vector<int> HostInput(N);
  std::iota(HostInput.begin(), HostInput.end(), 0);

  exp_ext::command_graph Graph1{
      Ctx, Dev, {exp_ext::property::graph::enable_native_recording{}}};
  exp_ext::command_graph Graph2{
      Ctx, Dev, {exp_ext::property::graph::enable_native_recording{}}};

  verifier.verify(EXECUTING, EXECUTING);

  Graph1.begin_recording(Queue1);
  verifier.verify(RECORDING, EXECUTING);

  exp_ext::enqueue_wait_event(Queue1, EventA, External);
  Queue1.memcpy(Staging, Data, N * sizeof(int));
  exp_ext::enqueue_signal_event(Queue1, EventB, External);

  Graph1.end_recording();

  Graph2.begin_recording(Queue2);
  verifier.verify(EXECUTING, RECORDING);

  exp_ext::enqueue_wait_event(Queue2, EventB, External);
  Queue2.parallel_for(range<1>{N},
                      [=](id<1> Idx) { Data[Idx] = Staging[Idx] + 1; });
  exp_ext::enqueue_signal_event(Queue2, EventA, External);

  Graph2.end_recording();
  verifier.verify(EXECUTING, EXECUTING);

  auto ExecutableGraph1 = Graph1.finalize();
  auto ExecutableGraph2 = Graph2.finalize();

  Queue1.memcpy(Data, HostInput.data(), N * sizeof(int));

  event GraphEvent2;
  for (unsigned Iter = 0; Iter < Iterations; Iter++) {
    Queue1.ext_oneapi_graph(ExecutableGraph1);
    GraphEvent2 = Queue2.submit(
        [&](handler &CGH) { CGH.ext_oneapi_graph(ExecutableGraph2); });
  }
  GraphEvent2.wait();

  std::vector<int> HostData(N);
  Queue2.memcpy(HostData.data(), Data, N * sizeof(int)).wait();
  for (size_t i = 0; i < N; i++) {
    assert(check_value(i, static_cast<int>(i + Iterations), HostData[i],
                       "HostData"));
  }

  free(Data, Ctx);
  free(Staging, Ctx);

  return 0;
}
