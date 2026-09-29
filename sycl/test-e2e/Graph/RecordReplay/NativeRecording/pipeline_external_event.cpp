// REQUIRES: level_zero_v2_adapter && arch-intel_gpu_bmg_g21

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// A native recording graph starts with a graph_external wait on an event,
// accumulates an input buffer into a device buffer, and ends with a
// graph_external signal of that same event. A separate in-order CopyQueue
// drives it for 100 replays with no host synchronization until the end: it
// uploads a new input and signals the event, the replay is submitted, then it
// waits on the event and copies the result out. The purpose is to overlap the
// copy engine with the compute engine on single compute engine hardware, so
// that only the event orders the work.
//
// Replay i adds i+1, so the copy after replay i must read the triangular number
// (i+1)(i+2)/2. A kernel which ran before its input landed, or a copy released
// by an earlier replay's signal, produces a different value.
//
// The test runs twice, with and without profiling enabled on the event, which
// gives the backend event a timestamp flag.

#include "../../graph_common.hpp"

#include <sycl/ext/oneapi/experimental/reusable_events.hpp>
#include <sycl/properties/all_properties.hpp>

#include <string>

constexpr int NumIters = 100;
constexpr size_t N = 4 * 1024;
constexpr size_t Bytes = N * sizeof(int);

void test(bool WithTimestamp) {
  device Dev;
  context Ctx{Dev};

  queue GraphQueue{Ctx, Dev, {property::queue::in_order{}}};
  // Never takes part in the recording.
  queue CopyQueue{Ctx, Dev, {property::queue::in_order{}}};

  QueueStateVerifier verifier(GraphQueue, CopyQueue);

  exp_ext::properties External{exp_ext::graph_external{}};
  exp_ext::properties Profiling{exp_ext::enable_profiling{true}};
  // Signaled by CopyQueue once the input is uploaded and waited on by the
  // graph, then signaled by the graph once the result is ready and waited on by
  // CopyQueue.
  event Event = WithTimestamp ? exp_ext::make_event(Ctx, Profiling)
                              : exp_ext::make_event(Ctx);

  int *Input = malloc_device<int>(N, Dev, Ctx);
  int *Data = malloc_device<int>(N, Dev, Ctx);
  int *HostInputs = malloc_host<int>(N * NumIters, Ctx);
  int *Snapshot = malloc_host<int>(N * NumIters, Ctx);
  assert(Input && Data && HostInputs && Snapshot);
  GraphQueue.memset(Data, 0, Bytes).wait();
  for (int I = 0; I < NumIters; ++I)
    std::fill(HostInputs + I * N, HostInputs + (I + 1) * N, I + 1);
  std::memset(Snapshot, 0, Bytes * NumIters);

  exp_ext::command_graph Graph{
      Ctx, Dev, {exp_ext::property::graph::enable_native_recording{}}};

  verifier.verify(EXECUTING, EXECUTING);

  Graph.begin_recording(GraphQueue);
  verifier.verify(RECORDING, EXECUTING);

  exp_ext::enqueue_wait_event(GraphQueue, Event, External);
  GraphQueue.parallel_for(range<1>{N},
                          [=](id<1> Idx) { Data[Idx] += Input[Idx]; });
  exp_ext::enqueue_signal_event(GraphQueue, Event, External);

  Graph.end_recording();
  verifier.verify(EXECUTING, EXECUTING);

  auto ExecutableGraph = Graph.finalize();

  // A wait on a reusable event binds to the most recent signal enqueued before
  // it, so the replay has to sit between CopyQueue's signal and CopyQueue's
  // wait: the graph's leading wait then consumes CopyQueue's signal, and
  // CopyQueue's wait consumes the graph's signal.
  for (int I = 0; I < NumIters; ++I) {
    CopyQueue.memcpy(Input, HostInputs + I * N, Bytes);
    exp_ext::enqueue_signal_event(CopyQueue, Event);

    GraphQueue.ext_oneapi_graph(ExecutableGraph);

    exp_ext::enqueue_wait_event(CopyQueue, Event);
    CopyQueue.memcpy(Snapshot + I * N, Data, Bytes);
  }

  // First and only host synchronization.
  CopyQueue.wait();
  GraphQueue.wait();

  for (int I = 0; I < NumIters; ++I) {
    const int Expected = (I + 1) * (I + 2) / 2;
    for (size_t J = 0; J < N; ++J) {
      assert(check_value(J, Expected, Snapshot[I * N + J],
                         "Snapshot after replay " + std::to_string(I + 1)));
    }
  }

  free(Input, Ctx);
  free(Data, Ctx);
  free(HostInputs, Ctx);
  free(Snapshot, Ctx);
}

int main() {
  test(/*WithTimestamp*/ false);
  test(/*WithTimestamp*/ true);
  return 0;
}
