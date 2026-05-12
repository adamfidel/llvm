// Tests that destruction callbacks registered on a graph are invoked when
// the graph is destroyed, and that args are kept alive by the graph even after
// the registering scope exits.

#include "../graph_common.hpp"

#include <sycl/properties/all_properties.hpp>

int main() {
  queue Queue{property::queue::in_order{}};

  bool FlagA = false;
  bool FlagB = false;

  {
#ifdef GRAPH_E2E_NATIVE_RECORDING
    exp_ext::command_graph Graph{
        Queue.get_context(),
        Queue.get_device(),
        {exp_ext::property::graph::enable_native_recording{}}};
#else
    exp_ext::command_graph Graph{Queue.get_context(), Queue.get_device()};
#endif

    const size_t N = 64;
    int *Data = malloc_device<int>(N, Queue);

    add_node(Graph, Queue, [&](handler &CGH) {
      CGH.parallel_for(range<1>{N},
                       [=](id<1> idx) { Data[idx] = static_cast<int>(idx); });
    });

    Graph.set_destruction_callback([](bool *Flag) { *Flag = true; }, &FlagA);

    // Registered from a nested scope — locals go out of scope but the graph's
    // stored tuple (via decay_t) retains valid copies of all args.
    {
      bool *LocalFlag = &FlagB;
      queue LocalQueue = Queue;
      int *LocalData = Data;
      Graph.set_destruction_callback(
          [](bool *Flag, queue Q, int *Ptr) {
            sycl::free(Ptr, Q);
            *Flag = true;
          },
          LocalFlag, LocalQueue, LocalData);
    }

    assert(!FlagA && "Callback A should not be invoked yet");
    assert(!FlagB && "Callback B should not be invoked yet");

    auto ExecGraph = Graph.finalize();
    Queue.submit([&](handler &CGH) { CGH.ext_oneapi_graph(ExecGraph); });
    Queue.wait();
  }

  assert(FlagA && "Callback A should have been invoked on destruction");
  assert(FlagB && "Callback B should have freed device memory on destruction");
  return 0;
}
