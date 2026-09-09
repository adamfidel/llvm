// REQUIRES: level_zero_v2_adapter && arch-intel_gpu_bmg_g21

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if level_zero %{%{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}

// Test a native recording graph with both an external output and an external
// input: the graph signals mid-replay, an outside queue observes that signal,
// uploads the data the rest of the graph consumes, and signals back. This is the
// only shape which shows the external signal is observable while the graph is
// still running rather than at graph completion.
//
// The outside work is a host to device copy: BMG has a single compute engine, so
// a kernel there would be serialized against the graph's kernels by the hardware
// whether or not the external events ordered it, whereas a copy runs on a copy
// engine and can genuinely overlap with the graph.
//
// This cannot pass until the external signal and wait UR entry points exist;
// today both are no-ops (see the "[TRACE] EXTERNAL_*" prints in queue_impl.cpp).

#include "../../graph_common.hpp"

#include <sycl/ext/oneapi/experimental/reusable_events.hpp>
#include <sycl/properties/all_properties.hpp>

int main() {
  device Dev;
  context Ctx{Dev};

  queue Queue1{Ctx, Dev, {property::queue::in_order{}}};
  // Never takes part in the recording.
  queue Queue2{Ctx, Dev, {property::queue::in_order{}}};

  QueueStateVerifier verifier(Queue1, Queue2);

  exp_ext::properties External{exp_ext::graph_external{}};
  event SignalEvent = exp_ext::make_event(Ctx);
  event WaitEvent = exp_ext::make_event(Ctx);

  const size_t N = 1024;
  int *Data = malloc_device<int>(N, Dev, Ctx);
  int *Extra = malloc_device<int>(N, Dev, Ctx);

  std::vector<int> HostExtra(N, 2);

  exp_ext::command_graph Graph{
      Ctx, Dev, {exp_ext::property::graph::enable_native_recording{}}};

  verifier.verify(EXECUTING, EXECUTING);

  Graph.begin_recording(Queue1);
  verifier.verify(RECORDING, EXECUTING);

  Queue1.parallel_for(range<1>{N},
                      [=](id<1> Idx) { Data[Idx] = static_cast<int>(Idx); });

  // Graph output, then graph input: the graph stops here until the outside queue
  // has uploaded Extra.
  exp_ext::enqueue_signal_event(Queue1, SignalEvent, External);
  exp_ext::enqueue_wait_event(Queue1, WaitEvent, External);

  Queue1.parallel_for(range<1>{N}, [=](id<1> Idx) { Data[Idx] += Extra[Idx]; });

  Graph.end_recording();
  verifier.verify(EXECUTING, EXECUTING);

  auto ExecutableGraph = Graph.finalize();

  std::vector<int> HostData(N);
  for (unsigned Iter = 0; Iter < Iterations; Iter++) {
    Queue2.fill(Extra, 0, N).wait();

    event GraphEvent = Queue1.submit(
        [&](handler &CGH) { CGH.ext_oneapi_graph(ExecutableGraph); });

    exp_ext::enqueue_wait_event(Queue2, SignalEvent);
    Queue2.memcpy(Extra, HostExtra.data(), N * sizeof(int));
    exp_ext::enqueue_signal_event(Queue2, WaitEvent);

    GraphEvent.wait();

    Queue2.memcpy(HostData.data(), Data, N * sizeof(int)).wait();
    for (size_t i = 0; i < N; i++) {
      assert(check_value(i, static_cast<int>(i) + 2, HostData[i], "HostData"));
    }
  }

  free(Data, Ctx);
  free(Extra, Ctx);

  return 0;
}
