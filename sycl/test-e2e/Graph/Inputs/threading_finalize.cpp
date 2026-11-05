// Tests finalizing the same modifiable graph concurrently from multiple
// threads. Each thread creates its own executable graph and submits it several
// times to a shared in-order queue, so the result is deterministic regardless
// of how the submissions interleave.

#include "../graph_threading.hpp"

#include <optional>

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

  using T = int;
  const unsigned NumThreads = numTestThreads();
  const unsigned SubmitsPerThread = 4;

  std::vector<T> DataA(Size), DataB(Size), DataC(Size);
  std::iota(DataA.begin(), DataA.end(), 1);
  std::iota(DataB.begin(), DataB.end(), 10);
  std::iota(DataC.begin(), DataC.end(), 1000);

  std::vector<T> ReferenceA(DataA), ReferenceB(DataB), ReferenceC(DataC);
  calculate_reference_data(NumThreads * SubmitsPerThread, Size, ReferenceA,
                           ReferenceB, ReferenceC);

  T *PtrA = malloc_device<T>(Size, Queue);
  T *PtrB = malloc_device<T>(Size, Queue);
  T *PtrC = malloc_device<T>(Size, Queue);

  Queue.copy(DataA.data(), PtrA, Size);
  Queue.copy(DataB.data(), PtrB, Size);
  Queue.copy(DataC.data(), PtrC, Size);
  Queue.wait_and_throw();

  Graph.begin_recording(Queue);
  run_kernels_usm(Queue, Size, PtrA, PtrB, PtrC);
  Graph.end_recording();

  std::vector<
      std::optional<exp_ext::command_graph<exp_ext::graph_state::executable>>>
      ExecGraphs(NumThreads);
  runConcurrently(NumThreads, [&](unsigned ThreadIndex) {
    auto &ExecGraph = ExecGraphs[ThreadIndex].emplace(Graph.finalize());
    for (unsigned I = 0; I < SubmitsPerThread; ++I)
      Queue.ext_oneapi_graph(ExecGraph);
  });

  Queue.copy(PtrA, DataA.data(), Size);
  Queue.copy(PtrB, DataB.data(), Size);
  Queue.copy(PtrC, DataC.data(), Size);
  Queue.wait_and_throw();

  bool Passed = true;
  for (size_t I = 0; I < Size; ++I) {
    Passed &= check_value(I, ReferenceA[I], DataA[I], "A");
    Passed &= check_value(I, ReferenceB[I], DataB[I], "B");
    Passed &= check_value(I, ReferenceC[I], DataC[I], "C");
  }

  free(PtrA, Queue);
  free(PtrB, Queue);
  free(PtrC, Queue);

  return Passed ? 0 : 1;
}
