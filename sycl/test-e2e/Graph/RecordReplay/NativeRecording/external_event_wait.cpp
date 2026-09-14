// REQUIRES: level_zero_v2_adapter && arch-intel_gpu_bmg_g21

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if level_zero %{%{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}

// Tests external event waits within the graph two ways:
// - An event is signaled prior to recording and never signaled again. The graph replay
// should honor this submission.
// - An event is never signaled prior to recording nor before the first replay. It is trivially complete
// on the first replay. Prior to the second replay, the signal is performed which should be honored.

#include "../../graph_common.hpp"

#include <sycl/ext/oneapi/experimental/reusable_events.hpp>
#include <sycl/properties/all_properties.hpp>

constexpr size_t N = 1024;
constexpr int HostOperand = 1;

using executable_graph =
    exp_ext::command_graph<exp_ext::graph_state::executable>;

executable_graph recordAddOperand(const context &Ctx, const device &Dev,
                                  queue &Q, event &ExtEvent, int *Data,
                                  int *Operand) {
  exp_ext::command_graph Graph{
      Ctx, Dev, {exp_ext::property::graph::enable_native_recording{}}};

  Graph.begin_recording(Q);
  exp_ext::enqueue_wait_event(Q, ExtEvent,
                              exp_ext::properties{exp_ext::graph_external{}});
  Q.parallel_for(range<1>{N}, [=](id<1> Idx) { Data[Idx] += Operand[0]; });
  Graph.end_recording();

  return Graph.finalize();
}

void checkData(queue &Q, int *Data, int Expected) {
  std::vector<int> HostData(N);
  Q.memcpy(HostData.data(), Data, N * sizeof(int)).wait();
  for (size_t i = 0; i < N; i++) {
    assert(check_value(i, Expected, HostData[i], "HostData"));
  }
}

// The event is signaled before the wait is recorded, so the single replay finds
// the wait satisfied and needs no signal of its own.
void signalBeforeRecording() {
  device Dev;
  context Ctx{Dev};

  queue Queue{Ctx, Dev, {property::queue::in_order{}}};
  // The producer never takes part in the recording.
  queue Producer{Ctx, Dev, {property::queue::in_order{}}};

  event ExtEvent = exp_ext::make_event(Ctx);

  int *Data = malloc_device<int>(N, Dev, Ctx);
  int *Operand = malloc_device<int>(1, Dev, Ctx);
  Queue.fill(Data, 0, N);
  Producer.fill(Operand, 0, 1).wait();

  // The operand upload is ordered ahead of the signal, so the recorded wait is
  // the only thing that can make it visible to the replay.
  Producer.memcpy(Operand, &HostOperand, sizeof(int));
  exp_ext::enqueue_signal_event(Producer, ExtEvent);

  executable_graph ExecGraph =
      recordAddOperand(Ctx, Dev, Queue, ExtEvent, Data, Operand);

  Queue.submit([&](handler &CGH) { CGH.ext_oneapi_graph(ExecGraph); }).wait();
  checkData(Producer, Data, 1);

  free(Data, Ctx);
  free(Operand, Ctx);
}

// On the first replay, the event has no signal and is thus trivially complete.
// On the second replay, the event is signaled after setting the operand.
void signalAfterFirstReplay() {
  device Dev;
  context Ctx{Dev};

  queue Queue{Ctx, Dev, {property::queue::in_order{}}};
  // The producer never takes part in the recording.
  queue Producer{Ctx, Dev, {property::queue::in_order{}}};

  event ExtEvent = exp_ext::make_event(Ctx);

  int *Data = malloc_device<int>(N, Dev, Ctx);
  int *Operand = malloc_device<int>(1, Dev, Ctx);
  Queue.fill(Data, 0, N);
  Queue.fill(Operand, 0, 1).wait();

  executable_graph ExecGraph =
      recordAddOperand(Ctx, Dev, Queue, ExtEvent, Data, Operand);

  // Event is trivially complete
  event FirstReplay =
      Queue.submit([&](handler &CGH) { CGH.ext_oneapi_graph(ExecGraph); });

  // Must depend on first submission to not taint the underlying data
  Producer.submit([&](handler &CGH) {
    CGH.depends_on(FirstReplay);
    CGH.memcpy(Operand, &HostOperand, sizeof(int));
  });
  exp_ext::enqueue_signal_event(Producer, ExtEvent);

  Queue.submit([&](handler &CGH) { CGH.ext_oneapi_graph(ExecGraph); }).wait();
  checkData(Producer, Data, 1);

  free(Data, Ctx);
  free(Operand, Ctx);
}

int main() {
  signalBeforeRecording();
  signalAfterFirstReplay();

  return 0;
}
