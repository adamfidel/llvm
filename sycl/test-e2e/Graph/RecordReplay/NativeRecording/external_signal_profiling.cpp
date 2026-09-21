// REQUIRES: level_zero_v2_adapter && arch-intel_gpu_bmg_g21
// REQUIRES: aspect-ext_oneapi_per_event_profiling

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if level_zero %{%{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}

// Test that two external events signalled either side of a kernel in a native
// recording graph carry profiling timestamps which bracket that kernel.

#include "../../graph_common.hpp"

#include <sycl/ext/oneapi/experimental/reusable_events.hpp>
#include <sycl/properties/all_properties.hpp>

int main() {
  device Dev;
  context Ctx{Dev};

  queue Queue{Ctx, Dev, {property::queue::in_order{}}};

  QueueStateVerifier verifier(Queue);

  exp_ext::properties EventProps{exp_ext::enable_profiling{true},
                                 exp_ext::graph_external{true}};
  exp_ext::properties External{exp_ext::graph_external{}};
  event Event1 = exp_ext::make_event(Ctx, EventProps);
  event Event2 = exp_ext::make_event(Ctx, EventProps);

  const size_t N = 1024;
  int *Data = malloc_device<int>(N, Dev, Ctx);

  exp_ext::command_graph Graph{
      Ctx, Dev, {exp_ext::property::graph::enable_native_recording{}}};

  verifier.verify(EXECUTING);

  Graph.begin_recording(Queue);
  verifier.verify(RECORDING);

  exp_ext::enqueue_signal_event(Queue, Event1, External);

  Queue.parallel_for(range<1>{N}, [=](id<1> Idx) { Data[Idx] += 1; });

  exp_ext::enqueue_signal_event(Queue, Event2, External);

  Graph.end_recording();
  verifier.verify(EXECUTING);

  auto ExecutableGraph = Graph.finalize();

  Queue.fill(Data, 0, N).wait();

  event GraphEvent = Queue.submit(
      [&](handler &CGH) { CGH.ext_oneapi_graph(ExecutableGraph); });
  Event2.wait();

  // A signal is an empty command, so a single event's start and end timestamps
  // may be identical. Compare across the two signals instead.
  auto Start =
      Event1.get_profiling_info<info::event_profiling::command_start>();
  auto End = Event2.get_profiling_info<info::event_profiling::command_start>();
  assert(End >= Start);
  assert(compareProfiling(Event1, Event2));

  GraphEvent.wait();

  std::vector<int> HostData(N);
  Queue.memcpy(HostData.data(), Data, N * sizeof(int)).wait();
  for (size_t i = 0; i < N; i++) {
    assert(check_value(i, 1, HostData[i], "HostData"));
  }

  free(Data, Ctx);

  return 0;
}
