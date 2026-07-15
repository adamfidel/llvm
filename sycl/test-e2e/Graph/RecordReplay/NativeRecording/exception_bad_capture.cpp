// REQUIRES: level_zero_v2_adapter && arch-intel_gpu_bmg_g21

// The spec-mandated errc::invalid is only reported once the underlying runtime
// exposes the granular graph capture error codes. Remove this XFAIL when that
// support lands.
// XFAIL: level_zero_v2_adapter

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if level_zero %{%{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}

// Tests that an operation which cannot be captured while a graph is being
// recorded throws errc::invalid. Exercises the UR
// UR_RESULT_ERROR_GRAPH_CAPTURE_UNSUPPORTED path handled when appending a
// command to the in-progress capture.
//
// The exact operation the driver rejects as capture-unsupported is device
// dependent. A regular kernel submission is used here as the representative
// command that flows through the command-buffer append path; substitute the
// specific unsupported operation once the runtime reports the error code.

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

  if (!expectException(
          [&]() {
            Queue.parallel_for(sycl::range<1>{N},
                               [=](sycl::id<1> Idx) { Data[Idx] = Idx; });
          },
          "unsupported operation during graph recording", errc::invalid)) {
    Graph.end_recording(Queue);
    free(Data, Queue);
    return 1;
  }

  Graph.end_recording(Queue);
  free(Data, Queue);

  return 0;
}
