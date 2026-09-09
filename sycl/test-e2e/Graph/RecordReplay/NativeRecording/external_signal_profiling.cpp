// REQUIRES: level_zero_v2_adapter && arch-intel_gpu_bmg_g21
// REQUIRES: aspect-ext_oneapi_per_event_profiling

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if level_zero %{%{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}

// Test that two external signals recorded around the work of a native recording
// graph carry timestamps which bracket that work, and that the timestamps are
// refreshed by each replay.
//
// Only external signals are timed here: an event whose backend handle was
// created by an external wait (event_impl::materializeExternalEvent) has no
// associated queue, so profiling queries on it are not well defined.
//
// This cannot pass until the external signal UR entry point exists; today the
// recorded signals are no-ops (see "[TRACE] EXTERNAL_SIGNAL" in queue_impl.cpp)
// so the host wait never returns.

#include "../../graph_common.hpp"

#include <sycl/ext/oneapi/experimental/reusable_events.hpp>
#include <sycl/properties/all_properties.hpp>

int main() {
  device Dev;
  context Ctx{Dev};

  queue Queue1{Ctx, Dev, {property::queue::in_order{}}};

  QueueStateVerifier verifier(Queue1);

  exp_ext::properties External{exp_ext::graph_external{}};
  exp_ext::properties ProfProps{exp_ext::enable_profiling{true}};
  event StartEvent = exp_ext::make_event(Ctx, ProfProps);
  event EndEvent = exp_ext::make_event(Ctx, ProfProps);

  const size_t N = 1024;
  int *Data = malloc_device<int>(N, Dev, Ctx);

  exp_ext::command_graph Graph{
      Ctx, Dev, {exp_ext::property::graph::enable_native_recording{}}};

  verifier.verify(EXECUTING);

  Graph.begin_recording(Queue1);
  verifier.verify(RECORDING);

  exp_ext::enqueue_signal_event(Queue1, StartEvent, External);

  for (unsigned i = 0; i < 3; i++) {
    Queue1.parallel_for(range<1>{N}, [=](id<1> Idx) { Data[Idx] += 1; });
  }

  exp_ext::enqueue_signal_event(Queue1, EndEvent, External);

  Graph.end_recording();
  verifier.verify(EXECUTING);

  auto ExecutableGraph = Graph.finalize();

  std::vector<int> HostData(N);
  uint64_t PreviousEnd = 0;
  for (unsigned Iter = 0; Iter < 2; Iter++) {
    Queue1.fill(Data, 0, N).wait();

    event GraphEvent = Queue1.submit(
        [&](handler &CGH) { CGH.ext_oneapi_graph(ExecutableGraph); });
    EndEvent.wait();

    // A signal is an empty command, so its start and end timestamps may be
    // identical. Compare the two signals rather than a single event's span.
    auto Start =
        StartEvent.get_profiling_info<info::event_profiling::command_end>();
    auto End =
        EndEvent.get_profiling_info<info::event_profiling::command_start>();
    assert(End >= Start);
    assert(compareProfiling(StartEvent, EndEvent));

    // The replays cannot overlap, so stale timestamps mean the events were not
    // re-armed by the second replay.
    assert(Start >= PreviousEnd);
    PreviousEnd = End;

    GraphEvent.wait();
    Queue1.memcpy(HostData.data(), Data, N * sizeof(int)).wait();
    for (size_t i = 0; i < N; i++) {
      assert(check_value(i, 3, HostData[i], "HostData"));
    }
  }

  free(Data, Ctx);

  return 0;
}
