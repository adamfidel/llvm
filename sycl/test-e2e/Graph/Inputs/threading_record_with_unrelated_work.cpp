// Tests that work submitted from another thread to a non-recording queue in the
// same context executes eagerly and is not captured while a graph is
// recording. Barriers keep the whole unrelated workload inside the recording
// window.

#include "../graph_threading.hpp"

int main() {
  queue RecordQueue{property::queue::in_order{}};
  queue WorkQueue{RecordQueue.get_context(), RecordQueue.get_device(),
                  property::queue::in_order{}};

#ifdef GRAPH_E2E_NATIVE_RECORDING
  exp_ext::command_graph Graph{
      RecordQueue.get_context(),
      RecordQueue.get_device(),
      {exp_ext::property::graph::enable_native_recording{}}};
#else
  exp_ext::command_graph Graph{RecordQueue.get_context(),
                               RecordQueue.get_device()};
#endif

  constexpr int RecordedIncrement = 7;
  std::vector<int> HostRecorded(Size);
  std::iota(HostRecorded.begin(), HostRecorded.end(), 0);
  int *Recorded = malloc_device<int>(Size, RecordQueue);
  RecordQueue.copy(HostRecorded.data(), Recorded, Size).wait_and_throw();

  int *Work = malloc_shared<int>(Size, WorkQueue);

  constexpr unsigned RecorderThread = 0;
  Barrier RecordingStarted{2};
  Barrier WorkDone{2};
  bool WorkPassed = true;

  auto RecordOrWork = [&](unsigned ThreadIndex) {
    if (ThreadIndex == RecorderThread) {
      Graph.begin_recording(RecordQueue);
      RecordingStarted.wait();
      RecordQueue.parallel_for(
          range<1>{Size}, [=](id<1> Id) { Recorded[Id] += RecordedIncrement; });
      WorkDone.wait();
      Graph.end_recording(RecordQueue);
      return;
    }

    RecordingStarted.wait();
    WorkPassed &= WorkQueue.ext_oneapi_get_state() == EXECUTING;

    WorkQueue.parallel_for(
        range<1>{Size}, [=](id<1> Id) { Work[Id] = static_cast<int>(Id[0]); });
    WorkQueue.submit([&](handler &CGH) {
      CGH.host_task([=] {
        for (size_t I = 0; I < Size; ++I)
          Work[I] += 1;
      });
    });
    WorkQueue.parallel_for(range<1>{Size}, [=](id<1> Id) { Work[Id] *= 3; });
    WorkQueue.wait_and_throw();

    for (size_t I = 0; I < Size; ++I) {
      const int Reference = (static_cast<int>(I) + 1) * 3;
      WorkPassed &= check_value(I, Reference, Work[I], "Work");
    }
    WorkPassed &= WorkQueue.ext_oneapi_get_state() == EXECUTING;
    WorkDone.wait();
  };
  runConcurrently(2, RecordOrWork, RecordingStarted, WorkDone);

  auto ExecGraph = Graph.finalize();
  for (unsigned I = 0; I < Iterations; ++I)
    RecordQueue.ext_oneapi_graph(ExecGraph);

  std::vector<int> Result(Size);
  RecordQueue.copy(Recorded, Result.data(), Size).wait_and_throw();

  bool Passed = WorkPassed;
  for (size_t I = 0; I < Size; ++I) {
    const int Reference =
        HostRecorded[I] + RecordedIncrement * static_cast<int>(Iterations);
    Passed &= check_value(I, Reference, Result[I], "Recorded");
  }

  free(Recorded, RecordQueue);
  free(Work, WorkQueue);

  return Passed ? 0 : 1;
}
