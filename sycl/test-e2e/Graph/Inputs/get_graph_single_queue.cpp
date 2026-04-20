// Test ext_oneapi_get_graph() with single queue

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

  const size_t N = 1024;
  int *Data = malloc_device<int>(N, Queue);

  Graph.begin_recording(Queue);

  auto RetrievedGraph1 = Queue.ext_oneapi_get_graph();
  assert(RetrievedGraph1 == Graph);

  Queue.submit([&](handler &CGH) {
    CGH.parallel_for(range<1>{N},
                     [=](id<1> idx) { Data[idx] = static_cast<int>(idx); });
  });

  auto RetrievedGraph2 = Queue.ext_oneapi_get_graph();
  assert(RetrievedGraph2 == Graph);

  Graph.end_recording(Queue);

  bool ExceptionThrown = false;
  try {
    Queue.ext_oneapi_get_graph();
  } catch (sycl::exception &e) {
    ExceptionThrown = true;
  }
  assert(ExceptionThrown && "Expected exception after end_recording");

  auto ExecutableGraph = Graph.finalize();
  Queue.submit([&](handler &CGH) { CGH.ext_oneapi_graph(ExecutableGraph); });
  Queue.wait();

  std::vector<int> HostData(N);
  Queue.memcpy(HostData.data(), Data, N * sizeof(int)).wait();

  for (size_t i = 0; i < N; i++) {
    assert(check_value(i, static_cast<int>(i), HostData[i], "HostData"));
  }

  free(Data, Queue);

  return 0;
}
