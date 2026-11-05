// Tests multiple threads concurrently submitting to the same recording in-order
// queue. The order in which threads' submissions interleave is undefined, so
// each thread only touches its own slice of the data. Each thread's chain of
// kernels is not commutative, so the in-order queue must also preserve the
// per-thread submission order for the result to be correct.

#include "../graph_threading.hpp"

constexpr int KernelsPerThread = 4;

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

  const unsigned NumThreads = numTestThreads();
  int *Data = malloc_device<int>(Size * NumThreads, Queue);

  Graph.begin_recording(Queue);

  Queue.parallel_for(range<1>{Size * NumThreads},
                     [=](id<1> Id) { Data[Id] = static_cast<int>(Id[0]); });

  runConcurrently(NumThreads, [&](unsigned ThreadIndex) {
    const int Thread = static_cast<int>(ThreadIndex);
    int *Slice = Data + ThreadIndex * Size;

    for (int Kernel = 0; Kernel < KernelsPerThread; ++Kernel)
      Queue.parallel_for(range<1>{Size}, [=](id<1> Id) {
        Slice[Id] = Slice[Id] * 2 + Thread + Kernel;
      });
  });

  bool Passed = Queue.ext_oneapi_get_state() == RECORDING;
  Graph.end_recording(Queue);
  Passed &= Queue.ext_oneapi_get_state() == EXECUTING;

#ifndef GRAPH_E2E_NATIVE_RECORDING
  const size_t ExpectedNodes = 1 + NumThreads * KernelsPerThread;
  Passed &= check_value(0, ExpectedNodes, Graph.get_nodes().size(), "Nodes");
#endif

  auto ExecGraph = Graph.finalize();
  for (unsigned I = 0; I < Iterations; ++I)
    Queue.ext_oneapi_graph(ExecGraph);

  std::vector<int> HostData(Size * NumThreads);
  Queue.copy(Data, HostData.data(), Size * NumThreads).wait_and_throw();

  for (unsigned Thread = 0; Thread < NumThreads; ++Thread) {
    for (size_t I = 0; I < Size; ++I) {
      const size_t Index = Thread * Size + I;
      int Reference = static_cast<int>(Index);
      for (int Kernel = 0; Kernel < KernelsPerThread; ++Kernel)
        Reference = Reference * 2 + static_cast<int>(Thread) + Kernel;
      Passed &= check_value(Index, Reference, HostData[Index], "Data");
    }
  }

  free(Data, Queue);

  return Passed ? 0 : 1;
}
