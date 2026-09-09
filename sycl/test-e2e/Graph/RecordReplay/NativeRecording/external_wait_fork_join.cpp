// REQUIRES: level_zero_v2_adapter && arch-intel_gpu_bmg_g21

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if level_zero %{%{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}

// Test that an external wait recorded into a native recording graph is a graph
// input: the whole graph, including a fork-join recorded behind the wait, only
// runs once the event is signaled from outside the graph. The buffer is zeroed
// before each replay and the input is produced after the graph is launched, so a
// dropped external wait gives a deterministic wrong result.
//
// The input is produced by a host to device copy: BMG has a single compute
// engine, so two kernels are serialized by the hardware whether or not they are
// ordered by an event, whereas a copy runs on a copy engine and can genuinely
// overlap with the graph's kernels.
//
// This cannot pass until the external wait UR entry point exists; today the
// recorded wait is a no-op (see "[TRACE] EXTERNAL_WAIT" in queue_impl.cpp).

#include "../../graph_common.hpp"

#include <sycl/ext/oneapi/experimental/reusable_events.hpp>
#include <sycl/properties/all_properties.hpp>

int main() {
  device Dev;
  context Ctx{Dev};

  queue Queue1{Ctx, Dev, {property::queue::in_order{}}};
  queue Queue2{Ctx, Dev, {property::queue::in_order{}}};
  // The producer never takes part in the recording.
  queue Queue3{Ctx, Dev, {property::queue::in_order{}}};

  QueueStateVerifier verifier(Queue1, Queue2, Queue3);

  exp_ext::properties External{exp_ext::graph_external{}};
  event ExtEvent = exp_ext::make_event(Ctx);

  const size_t N = 1024;
  int *Data = malloc_device<int>(N, Dev, Ctx);

  std::vector<int> HostInput(N);
  std::iota(HostInput.begin(), HostInput.end(), 0);

  exp_ext::command_graph Graph{
      Ctx, Dev, {exp_ext::property::graph::enable_native_recording{}}};

  verifier.verify(EXECUTING, EXECUTING, EXECUTING);

  Graph.begin_recording(Queue1);
  verifier.verify(RECORDING, EXECUTING, EXECUTING);

  exp_ext::enqueue_wait_event(Queue1, ExtEvent, External);

  event Fork =
      Queue1.parallel_for(range<1>{N}, [=](id<1> Idx) { Data[Idx] *= 2; });

  event Branch = Queue2.submit([&](handler &CGH) {
    CGH.depends_on(Fork);
    CGH.parallel_for(range<1>{N}, [=](id<1> Idx) { Data[Idx] += 1; });
  });
  verifier.verify(RECORDING, RECORDING, EXECUTING);

  Queue1.submit([&](handler &CGH) {
    CGH.depends_on(Branch);
    CGH.parallel_for(range<1>{N}, [=](id<1> Idx) { Data[Idx] *= 3; });
  });

  Graph.end_recording();
  verifier.verify(EXECUTING, EXECUTING, EXECUTING);

  auto ExecutableGraph = Graph.finalize();

  std::vector<int> HostData(N);
  for (unsigned Iter = 0; Iter < Iterations; Iter++) {
    Queue3.fill(Data, 0, N).wait();

    event GraphEvent = Queue1.submit(
        [&](handler &CGH) { CGH.ext_oneapi_graph(ExecutableGraph); });

    // Produce the graph's input only after the graph has been launched.
    Queue3.memcpy(Data, HostInput.data(), N * sizeof(int));
    exp_ext::enqueue_signal_event(Queue3, ExtEvent);

    GraphEvent.wait();

    Queue3.memcpy(HostData.data(), Data, N * sizeof(int)).wait();
    for (size_t i = 0; i < N; i++) {
      int Expected = (static_cast<int>(i) * 2 + 1) * 3;
      assert(check_value(i, Expected, HostData[i], "HostData"));
    }
  }

  free(Data, Ctx);

  return 0;
}
