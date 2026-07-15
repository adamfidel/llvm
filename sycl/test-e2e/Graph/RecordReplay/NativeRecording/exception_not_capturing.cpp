// REQUIRES: level_zero_v2_adapter && arch-intel_gpu_bmg_g21

// The spec-mandated errc::invalid is only reported once the underlying runtime
// exposes the granular graph capture error codes. Remove this XFAIL when that
// support lands.
// XFAIL: level_zero_v2_adapter

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if level_zero %{%{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}

// Tests that ending a recording on a queue that is not currently capturing to
// the graph throws errc::invalid. Exercises the UR
// UR_RESULT_ERROR_COMMAND_LIST_NOT_CAPTURING path.

#include "../../graph_common.hpp"

#include <sycl/properties/all_properties.hpp>

int main() {
  queue Queue{property::queue::in_order{}};

  exp_ext::command_graph Graph{
      Queue.get_context(),
      Queue.get_device(),
      {exp_ext::property::graph::enable_native_recording{}}};

  // The queue was never put into recording mode for this graph, so ending the
  // recording is invalid.
  if (!expectException([&]() { Graph.end_recording(Queue); },
                       "end_recording on a non-recording queue",
                       errc::invalid)) {
    return 1;
  }

  return 0;
}
