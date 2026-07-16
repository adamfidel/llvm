// Tests that a restricted syclex::host_task() submitted through the
// handler + submit_with_event path can be recorded into a SYCL Graph and
// participate in event-based dependencies across two in-order queues:
//   - Fork: the host task produces an event that a second queue consumes,
//           transitioning that queue into recording.
//   - Join: a later host task consumes an event from the second queue,
//           ordering itself after that queue's work.

#include "../../graph_common.hpp"

#include <sycl/ext/oneapi/experimental/enqueue_functions.hpp>
#include <sycl/properties/all_properties.hpp>

namespace syclex = sycl::ext::oneapi::experimental;

constexpr size_t N = 1024;

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

  uint32_t *Data = malloc_shared<uint32_t>(N, Queue1);
  std::fill(Data, Data + N, 0);

  Graph.begin_recording(Queue1);

  // Q1 kernel: accumulate index so replays produce distinct results.
  Queue1.submit([&](handler &CGH) {
    CGH.parallel_for(range<1>{N}, [=](id<1> idx) {
      Data[idx] += static_cast<uint32_t>(idx[0]) + 1;
    });
  });

  // Fork: host task on Q1 doubles the data and signals ForkEvent.
  auto ForkEvent = syclex::submit_with_event(Queue1, [&](handler &CGH) {
    syclex::host_task(CGH, [=] {
      for (size_t i = 0; i < N; i++)
        Data[i] *= 2;
    });
  });

  // Q2 consumes ForkEvent, which transitions it into recording (fork).
  auto Q2Event = syclex::submit_with_event(Queue2, [&](handler &CGH) {
    CGH.depends_on(ForkEvent);
    CGH.parallel_for(range<1>{N}, [=](id<1> idx) { Data[idx] += 10; });
  });

  assert(Queue2.ext_oneapi_get_state() == RECORDING &&
         "host task event did not fork the second queue into recording");

  // Join: host task on Q1 waits on Q2Event before adding to the data.
  syclex::submit_with_event(Queue1, [&](handler &CGH) {
    CGH.depends_on(Q2Event);
    syclex::host_task(CGH, [=] {
      for (size_t i = 0; i < N; i++)
        Data[i] += 100;
    });
  });

  Graph.end_recording();

  auto ExecutableGraph = Graph.finalize();

  // Per replay, starting from D: D -> 2 * (D + (i + 1)) + 10 + 100.
  Queue1.submit([&](handler &CGH) { CGH.ext_oneapi_graph(ExecutableGraph); });
  Queue1.wait();

  for (size_t i = 0; i < N; i++) {
    uint32_t Expected = 2 * (static_cast<uint32_t>(i) + 1) + 110;
    assert(check_value(i, Expected, Data[i], "Data"));
  }

  Queue1.submit([&](handler &CGH) { CGH.ext_oneapi_graph(ExecutableGraph); });
  Queue1.wait();

  for (size_t i = 0; i < N; i++) {
    uint32_t Expected = 6 * (static_cast<uint32_t>(i) + 1) + 330;
    assert(check_value(i, Expected, Data[i], "Data"));
  }

  free(Data, Queue1);
  return 0;
}
