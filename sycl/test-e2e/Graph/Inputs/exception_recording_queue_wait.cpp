// Tests that waiting on a Queue in recording mode throws.

#include "../graph_common.hpp"

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
  Graph.begin_recording(Queue);

  if (!expectException([&]() { Queue.wait(); },
                       "queue wait during graph recording", errc::invalid)) {
    return 1;
  }

  Graph.end_recording();

  return 0;
}
