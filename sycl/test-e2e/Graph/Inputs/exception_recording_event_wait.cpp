// Tests that waiting on an event signaled during graph recording
// throws.

#include "../graph_common.hpp"

#include <sycl/properties/all_properties.hpp>

int main() {
  queue Queue{property::queue::in_order{}};

#ifdef GRAPH_E2E_NATIVE_RECORDING
  exp_ext::command_graph Graph{
      Queue.get_context(),
      Queue.get_device(),
      {exp_ext::property::graph::enable_native_recording{}}};
  // In native recording the error originates in the runtime and is reported as
  // errc::runtime, whereas traditional recording rejects the wait at the SYCL
  // level with errc::invalid.
  constexpr errc ExpectedCode = errc::runtime;
#else
  exp_ext::command_graph Graph{Queue.get_context(), Queue.get_device()};
  constexpr errc ExpectedCode = errc::invalid;
#endif

  Graph.begin_recording(Queue);

  auto GraphEvent = Queue.single_task([]() {});

  if (!expectException([&]() { GraphEvent.wait(); },
                       "event wait during graph recording", ExpectedCode)) {
    return 1;
  }

  Graph.end_recording();

  return 0;
}
