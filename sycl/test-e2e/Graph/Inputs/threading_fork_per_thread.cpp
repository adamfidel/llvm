// Tests recording a fork-join graph where each forked branch is recorded from
// its own thread. The primary queue records a fork point, every thread then
// concurrently submits a dependent chain of kernels to its own in-order queue,
// which transitions that queue to recording. The primary queue joins all
// branches.

#include "../graph_threading.hpp"

constexpr int KernelsPerThread = 4;

int main() {
  queue Primary{property::queue::in_order{}};

#ifdef GRAPH_E2E_NATIVE_RECORDING
  exp_ext::command_graph Graph{
      Primary.get_context(),
      Primary.get_device(),
      {exp_ext::property::graph::enable_native_recording{}}};
#else
  exp_ext::command_graph Graph{Primary.get_context(), Primary.get_device()};
#endif

  const unsigned NumForks = numTestThreads();
  std::vector<queue> ForkQueues = makeInOrderQueues(Primary, NumForks);

  int *Data = malloc_device<int>(Size * NumForks, Primary);
  int *Sum = malloc_device<int>(Size, Primary);

  Graph.begin_recording(Primary);

  event Fork = Primary.parallel_for(range<1>{Size * NumForks}, [=](id<1> Id) {
    Data[Id] = static_cast<int>(Id[0]);
  });

  std::vector<event> Joins(NumForks);
  runConcurrently(NumForks, [&](unsigned ThreadIndex) {
    queue &ForkQueue = ForkQueues[ThreadIndex];
    const int Branch = static_cast<int>(ThreadIndex);
    int *BranchData = Data + ThreadIndex * Size;

    auto Step = [=](int Kernel) {
      return [=](id<1> Id) {
        BranchData[Id] = BranchData[Id] * 2 + Branch + Kernel;
      };
    };

    event Last = ForkQueue.parallel_for(range<1>{Size}, Fork, Step(0));
    for (int Kernel = 1; Kernel < KernelsPerThread; ++Kernel)
      Last = ForkQueue.parallel_for(range<1>{Size}, Step(Kernel));
    Joins[ThreadIndex] = Last;
  });

  bool Passed = checkQueueStates(ForkQueues, RECORDING);

  Primary.parallel_for(range<1>{Size}, Joins, [=](id<1> Id) {
    int Total = 0;
    for (unsigned Branch = 0; Branch < NumForks; ++Branch)
      Total += Data[Branch * Size + Id[0]];
    Sum[Id] = Total;
  });

  Graph.end_recording();

  Passed &= checkQueueStates(ForkQueues, EXECUTING);
  Passed &= Primary.ext_oneapi_get_state() == EXECUTING;

  auto ExecGraph = Graph.finalize();
  for (unsigned I = 0; I < Iterations; ++I)
    Primary.ext_oneapi_graph(ExecGraph);

  std::vector<int> HostSum(Size);
  Primary.copy(Sum, HostSum.data(), Size).wait_and_throw();

  for (size_t I = 0; I < Size; ++I) {
    int Reference = 0;
    for (unsigned Branch = 0; Branch < NumForks; ++Branch) {
      int Value = static_cast<int>(Branch * Size + I);
      for (int Kernel = 0; Kernel < KernelsPerThread; ++Kernel)
        Value = Value * 2 + static_cast<int>(Branch) + Kernel;
      Reference += Value;
    }
    Passed &= check_value(I, Reference, HostSum[I], "Sum");
  }

  free(Data, Primary);
  free(Sum, Primary);

  return Passed ? 0 : 1;
}
