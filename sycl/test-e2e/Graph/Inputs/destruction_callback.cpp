// Tests destruction callbacks: resource cleanup on graph destruction, and
// copy/move constraint validation (CopyConstructible required,
// MoveConstructible as rvalue optimization, assignment not required).

#include "../graph_common.hpp"

#include <sycl/properties/all_properties.hpp>

struct Tracker {
  bool *CopiedFlag;
  bool *MovedFlag;
  int Value;

  Tracker(int v, bool *copied, bool *moved)
      : CopiedFlag(copied), MovedFlag(moved), Value(v) {}
  Tracker(const Tracker &Other)
      : CopiedFlag(Other.CopiedFlag), MovedFlag(Other.MovedFlag),
        Value(Other.Value) {
    *CopiedFlag = true;
  }
  Tracker(Tracker &&Other)
      : CopiedFlag(Other.CopiedFlag), MovedFlag(Other.MovedFlag),
        Value(Other.Value) {
    *MovedFlag = true;
  }
  Tracker &operator=(const Tracker &) = delete;
  Tracker &operator=(Tracker &&) = delete;
};

int main() {
  queue Queue{property::queue::in_order{}};

  int ObservedValueLvalue = 0;
  int ObservedValueRvalue = 0;
  int ObservedN = 0;

  const size_t N = 64;
  int *Data = malloc_device<int>(N, Queue);

  {
#ifdef GRAPH_E2E_NATIVE_RECORDING
    exp_ext::command_graph Graph{
        Queue.get_context(),
        Queue.get_device(),
        {exp_ext::property::graph::enable_native_recording{}}};
#else
    exp_ext::command_graph Graph{Queue.get_context(), Queue.get_device()};
#endif

    add_node(Graph, Queue, [&](handler &CGH) {
      CGH.parallel_for(range<1>{N},
                       [=](id<1> idx) { Data[idx] = static_cast<int>(idx); });
    });

    // Resource cleanup: free device memory on graph destruction.
    // Also verifies lvalue args are copied (not captured): N is passed by value
    // then immediately mutated — the callback must see the original.
    int **DataPtr = &Data;
    int FreeCopyN = static_cast<int>(N);
    Graph.set_destruction_callback(
        [](int **Ptr, queue Q, int Count, int *Out) {
          sycl::free(*Ptr, Q);
          *Ptr = nullptr;
          *Out = Count;
        },
        DataPtr, Queue, FreeCopyN, &ObservedN);
    FreeCopyN = 0;

    // Lvalue arg: must be copied into the graph's stored tuple.
    bool LvalueCopied = false, LvalueMoved = false;
    Tracker LvalueTracker{42, &LvalueCopied, &LvalueMoved};
    Graph.set_destruction_callback([](Tracker T, int *Out) { *Out = T.Value; },
                                   LvalueTracker, &ObservedValueLvalue);
    assert(LvalueCopied && "Lvalue arg should be copied");

    // Rvalue arg: move-constructible optimization should kick in.
    bool RvalueCopied = false, RvalueMoved = false;
    Tracker RvalueTracker{99, &RvalueCopied, &RvalueMoved};
    Graph.set_destruction_callback([](Tracker T, int *Out) { *Out = T.Value; },
                                   std::move(RvalueTracker),
                                   &ObservedValueRvalue);
    assert(RvalueMoved && !RvalueCopied &&
           "Rvalue arg should be moved, not copied");

    assert(Data != nullptr && "Free callback should not be invoked yet");

    auto ExecGraph = Graph.finalize();
    Queue.submit([&](handler &CGH) { CGH.ext_oneapi_graph(ExecGraph); });
    Queue.wait();
  }

  assert(Data == nullptr && "Free callback should have nulled the pointer");
  assert(ObservedValueLvalue == 42 &&
         "Lvalue callback should see original value");
  assert(ObservedValueRvalue == 99 &&
         "Rvalue callback should see original value");
  assert(ObservedN == 64 &&
         "Free callback should see original N, not mutated value");
  return 0;
}
