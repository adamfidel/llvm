// Tests independent graphs being created, recorded, finalized, executed and
// destroyed concurrently, one per thread, on queues sharing a context. A
// barrier after begin_recording makes every recording active at the same time,
// so work captured into the wrong graph shows up as a wrong result.

#include "../graph_threading.hpp"

int main() {
  queue Queue;

  const unsigned NumThreads = numTestThreads();
  std::vector<queue> Queues = makeInOrderQueues(Queue, NumThreads);
  Barrier AllRecording{NumThreads};
  std::vector<int> Passed(NumThreads, 0);

  runConcurrently(NumThreads, [&](unsigned ThreadIndex) {
    queue &ThreadQueue = Queues[ThreadIndex];
    const int Increment = static_cast<int>(ThreadIndex) + 1;

#ifdef GRAPH_E2E_NATIVE_RECORDING
    exp_ext::command_graph Graph{
        ThreadQueue.get_context(),
        ThreadQueue.get_device(),
        {exp_ext::property::graph::enable_native_recording{}}};
#else
    exp_ext::command_graph Graph{ThreadQueue.get_context(),
                                 ThreadQueue.get_device()};
#endif

    std::vector<int> HostData(Size);
    std::iota(HostData.begin(), HostData.end(), 0);
    int *Data = malloc_device<int>(Size, ThreadQueue);
    ThreadQueue.copy(HostData.data(), Data, Size).wait_and_throw();

    Graph.begin_recording(ThreadQueue);
    AllRecording.wait();

    ThreadQueue.parallel_for(range<1>{Size},
                             [=](id<1> Id) { Data[Id] += Increment; });
    ThreadQueue.parallel_for(range<1>{Size}, [=](id<1> Id) { Data[Id] *= 2; });

    bool ThreadPassed = ThreadQueue.ext_oneapi_get_state() == RECORDING;
    Graph.end_recording(ThreadQueue);
    ThreadPassed &= ThreadQueue.ext_oneapi_get_state() == EXECUTING;

    auto ExecGraph = Graph.finalize();
    for (unsigned I = 0; I < Iterations; ++I)
      ThreadQueue.ext_oneapi_graph(ExecGraph);

    std::vector<int> Result(Size);
    ThreadQueue.copy(Data, Result.data(), Size).wait_and_throw();

    for (size_t I = 0; I < Size; ++I) {
      int Reference = HostData[I];
      for (unsigned Iter = 0; Iter < Iterations; ++Iter)
        Reference = (Reference + Increment) * 2;
      ThreadPassed &= check_value(I, Reference, Result[I], "Data");
    }

    free(Data, ThreadQueue);
    Passed[ThreadIndex] = ThreadPassed;
  });

  const bool AllPassed =
      std::all_of(Passed.begin(), Passed.end(), [](int P) { return P; });
  return AllPassed ? 0 : 1;
}
