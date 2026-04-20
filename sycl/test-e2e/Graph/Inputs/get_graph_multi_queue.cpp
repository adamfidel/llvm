// Test ext_oneapi_get_graph() with multiple queues (fork-join pattern)

#include "../graph_common.hpp"
#include <sycl/properties/all_properties.hpp>

int main() {
  device Dev;
  context Ctx{Dev};

  queue Queue1{Ctx, Dev, {property::queue::in_order{}}};
  queue Queue2{Ctx, Dev, {property::queue::in_order{}}};

#ifdef GRAPH_E2E_NATIVE_RECORDING
  exp_ext::command_graph Graph{
      Ctx, Dev, {exp_ext::property::graph::enable_native_recording{}}};
#else
  exp_ext::command_graph Graph{Ctx, Dev};
#endif

  const size_t N = 1024;
  int *DataA = malloc_device<int>(N, Dev, Ctx);
  int *DataB = malloc_device<int>(N, Dev, Ctx);
  int *Result = malloc_device<int>(1, Dev, Ctx);

  Graph.begin_recording(Queue1);

  auto RetrievedGraph1 = Queue1.ext_oneapi_get_graph();
  assert(RetrievedGraph1 == Graph);

  auto Event1 = Queue1.parallel_for(range<1>{N}, [=](item<1> idx) {
    DataA[idx] = static_cast<int>(idx);
  });

  auto Event2 = Queue2.parallel_for(range<1>{N}, {Event1}, [=](item<1> idx) {
    DataB[idx] = static_cast<int>(idx) * 2;
  });

  auto RetrievedGraph2 = Queue2.ext_oneapi_get_graph();
  assert(RetrievedGraph2 == Graph);

  Queue1.single_task({Event2}, [=]() {
    int sum = 0;
    for (size_t i = 0; i < N; i++) {
      sum += DataA[i] + DataB[i];
    }
    Result[0] = sum;
  });

  Graph.end_recording();

  assert(expectException([&]() { Queue1.ext_oneapi_get_graph(); },
                         "Queue1.ext_oneapi_get_graph() after end_recording",
                         sycl::errc::invalid) &&
         "Expected exception on Queue1 after end_recording");

  assert(expectException([&]() { Queue2.ext_oneapi_get_graph(); },
                         "Queue2.ext_oneapi_get_graph() after end_recording",
                         sycl::errc::invalid) &&
         "Expected exception on Queue2 after end_recording");

  auto ExecutableGraph = Graph.finalize();
  Queue1.submit([&](handler &CGH) { CGH.ext_oneapi_graph(ExecutableGraph); });
  Queue1.wait();

  int HostResult = 0;
  Queue1.memcpy(&HostResult, Result, sizeof(int)).wait();

  int Expected = 0;
  for (size_t i = 0; i < N; i++) {
    Expected += i + (i * 2);
  }

  assert(check_value(0, Expected, HostResult, "Result"));

  free(DataA, Ctx);
  free(DataB, Ctx);
  free(Result, Ctx);

  return 0;
}
