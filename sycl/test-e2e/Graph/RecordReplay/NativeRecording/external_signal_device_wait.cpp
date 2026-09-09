// REQUIRES: level_zero_v2_adapter && arch-intel_gpu_bmg_g21

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if level_zero %{%{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}

// Test that an external signal recorded into a native recording graph is a graph
// output which can be consumed by a device-side wait on a queue that never took
// part in the recording. The consumer is ordered against the graph only by the
// external event: the test waits on the consumer queue, not on the graph event.
//
// The consumer is a device to host copy: BMG has a single compute engine, so a
// consumer kernel would be serialized after the graph's kernels by the hardware
// whether or not the external event ordered it, whereas a copy runs on a copy
// engine and can genuinely overlap with the graph. The graph's tail writes a
// separate buffer so the copy only depends on the external event, not on where
// the graph has got to.
//
// This cannot pass until the external signal UR entry point exists; today the
// recorded signal is a no-op (see "[TRACE] EXTERNAL_SIGNAL" in queue_impl.cpp)
// so the consumer's wait never completes.

#include "../../graph_common.hpp"

#include <sycl/ext/oneapi/experimental/reusable_events.hpp>
#include <sycl/properties/all_properties.hpp>

int main() {
  device Dev;
  context Ctx{Dev};

  queue Queue1{Ctx, Dev, {property::queue::in_order{}}};
  // The consumer never takes part in the recording.
  queue Queue2{Ctx, Dev, {property::queue::in_order{}}};

  QueueStateVerifier verifier(Queue1, Queue2);

  exp_ext::properties External{exp_ext::graph_external{}};
  event ExtEvent = exp_ext::make_event(Ctx);

  const size_t N = 1024;
  int *Data = malloc_device<int>(N, Dev, Ctx);
  int *Tail = malloc_device<int>(N, Dev, Ctx);

  exp_ext::command_graph Graph{
      Ctx, Dev, {exp_ext::property::graph::enable_native_recording{}}};

  verifier.verify(EXECUTING, EXECUTING);

  Graph.begin_recording(Queue1);
  verifier.verify(RECORDING, EXECUTING);

  Queue1.parallel_for(range<1>{N}, [=](id<1> Idx) {
    Data[Idx] = static_cast<int>(Idx) * 2;
  });

  exp_ext::enqueue_signal_event(Queue1, ExtEvent, External);

  // Recorded after the signal, so the signal is observable before the graph
  // completes. Only reads Data.
  Queue1.parallel_for(range<1>{N},
                      [=](id<1> Idx) { Tail[Idx] = Data[Idx] + 1; });

  Graph.end_recording();
  verifier.verify(EXECUTING, EXECUTING);

  auto ExecutableGraph = Graph.finalize();

  std::vector<int> HostData(N);
  for (unsigned Iter = 0; Iter < Iterations; Iter++) {
    Queue2.fill(Data, 0, N);
    Queue2.fill(Tail, 0, N);
    Queue2.wait();

    event GraphEvent = Queue1.submit(
        [&](handler &CGH) { CGH.ext_oneapi_graph(ExecutableGraph); });

    exp_ext::enqueue_wait_event(Queue2, ExtEvent);
    Queue2.memcpy(HostData.data(), Data, N * sizeof(int));
    // Deliberately not GraphEvent.wait(): the external event is the only thing
    // ordering the copy against the graph.
    Queue2.wait();

    for (size_t i = 0; i < N; i++) {
      assert(check_value(i, static_cast<int>(i) * 2, HostData[i], "HostData"));
    }

    GraphEvent.wait();
    Queue2.memcpy(HostData.data(), Tail, N * sizeof(int)).wait();
    for (size_t i = 0; i < N; i++) {
      assert(
          check_value(i, static_cast<int>(i) * 2 + 1, HostData[i], "HostData"));
    }
  }

  free(Data, Ctx);
  free(Tail, Ctx);

  return 0;
}
