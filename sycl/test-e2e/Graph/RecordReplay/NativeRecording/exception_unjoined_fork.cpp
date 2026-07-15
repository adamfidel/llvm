// REQUIRES: level_zero_v2_adapter && arch-intel_gpu_bmg_g21

// The error is only reported once the underlying runtime detects and rejects
// the unjoined forks at end of capture. Remove this XFAIL when that support
// lands.
// XFAIL: level_zero_v2_adapter

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if level_zero %{%{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}

// Tests that ending a recording with forked branches that were not rejoined
// throws errc::runtime. Exercises the UR UR_RESULT_ERROR_GRAPH_UNJOINED_FORKS
// path handled on the end-capture call.
//
// A fork is created when work branches off the recorded stream without being
// joined back before end_recording. The precise operation that produces an
// unjoined fork is device dependent; an event-based branch is used here as the
// representative case.

#include "../../graph_common.hpp"

#include <sycl/properties/all_properties.hpp>

int main() {
  queue Queue{property::queue::in_order{}};

  exp_ext::command_graph Graph{
      Queue.get_context(),
      Queue.get_device(),
      {exp_ext::property::graph::enable_native_recording{}}};

  constexpr size_t N = 1024;
  int *Data = malloc_device<int>(N, Queue);

  Graph.begin_recording(Queue);

  // Create a branch off the recorded stream that is never joined back before
  // ending the recording.
  auto ForkEvent = Queue.parallel_for(
      sycl::range<1>{N}, [=](sycl::id<1> Idx) { Data[Idx] = Idx; });
  Queue.submit([&](handler &CGH) {
    CGH.depends_on(ForkEvent);
    CGH.parallel_for(sycl::range<1>{N},
                     [=](sycl::id<1> Idx) { Data[Idx] += 1; });
  });

  if (!expectException([&]() { Graph.end_recording(Queue); },
                       "end_recording with unjoined forks", errc::runtime)) {
    free(Data, Queue);
    return 1;
  }

  free(Data, Queue);

  return 0;
}
