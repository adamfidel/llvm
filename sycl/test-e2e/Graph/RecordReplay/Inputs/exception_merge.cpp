// Tests that an illegal attempt to merge two graph recordings throws
// errc::invalid. Exercises the UR UR_RESULT_ERROR_GRAPH_CAPTURE_MERGE_ATTEMPT
// path handled on the command-buffer append call.
//
// A merge attempt arises when work already being captured by one graph is
// pulled into the capture of a second graph via a cross-graph dependency.

#include "../../graph_common.hpp"

#include <sycl/properties/all_properties.hpp>

int main() {
  queue QueueA{property::queue::in_order{}};
  queue QueueB{QueueA.get_context(), QueueA.get_device(),
               property::queue::in_order{}};

#ifdef GRAPH_E2E_NATIVE_RECORDING
  exp_ext::command_graph GraphA{
      QueueA.get_context(),
      QueueA.get_device(),
      {exp_ext::property::graph::enable_native_recording{}}};
  exp_ext::command_graph GraphB{
      QueueB.get_context(),
      QueueB.get_device(),
      {exp_ext::property::graph::enable_native_recording{}}};
#else
  exp_ext::command_graph GraphA{QueueA.get_context(), QueueA.get_device()};
  exp_ext::command_graph GraphB{QueueB.get_context(), QueueB.get_device()};
#endif

  constexpr size_t N = 1024;
  int *Data = malloc_device<int>(N, QueueA);

  GraphA.begin_recording(QueueA);
  GraphB.begin_recording(QueueB);

  auto EventA = QueueA.parallel_for(sycl::range<1>{N},
                                    [=](sycl::id<1> Idx) { Data[Idx] = Idx; });

  // Recording work on QueueB that depends on QueueA's in-progress capture
  // would merge the two separate recordings, which is illegal.
  if (!expectException(
          [&]() {
            QueueB.submit([&](handler &CGH) {
              CGH.depends_on(EventA);
              CGH.parallel_for(sycl::range<1>{N},
                               [=](sycl::id<1> Idx) { Data[Idx] += 1; });
            });
          },
          "merging two graph recordings", errc::invalid)) {
    GraphA.end_recording(QueueA);
    GraphB.end_recording(QueueB);
    free(Data, QueueA);
    return 1;
  }

  GraphA.end_recording(QueueA);
  GraphB.end_recording(QueueB);
  free(Data, QueueA);

  return 0;
}
