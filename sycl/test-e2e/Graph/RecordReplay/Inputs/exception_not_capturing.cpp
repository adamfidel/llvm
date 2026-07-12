// Tests that ending a recording on a queue that is not currently capturing to
// the graph throws errc::invalid. Exercises the UR
// UR_RESULT_ERROR_COMMAND_LIST_NOT_CAPTURING path.

#include "../../graph_common.hpp"

#include <sycl/properties/all_properties.hpp>

int main() {
  queue Queue{property::queue::in_order{}};

#ifdef GRAPH_E2E_NATIVE_RECORDING
  exp_ext::command_graph Graph{
      Queue.get_context(),
      Queue.get_device(),
      {exp_ext::property::graph::enable_native_recording{}}};
#else
  exp_ext::command_graph Graph{Queue.get_context(), Queue.get_device()};
#endif

  // The queue was never put into recording mode for this graph, so ending the
  // recording is invalid.
  if (!expectException([&]() { Graph.end_recording(Queue); },
                       "end_recording on a non-recording queue",
                       errc::invalid)) {
    return 1;
  }

  return 0;
}
