// REQUIRES: level_zero_v2_adapter && (arch-intel_gpu_bmg_g21 || arch-intel_gpu_bmg_g31 || arch-intel_gpu_cri)
// REQUIRES-INTEL-DRIVER: lin: 40986

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if level_zero %{%{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=%{leak_check_not} %}

// Test that a reusable event transitions between graph internal and eager
// states, or between two concurrently recording graphs, based on the queue it
// is most recently signaled on:
//   1. Signaling the event on a queue recording graph 1 makes it internal to
//      graph 1, where it is used for a fork-join.
//   2. Signaling the event on another queue moves it out of graph 1. With a
//      single graph that queue is not recording, so the event becomes eager.
//      With multiple graphs that queue records graph 2, so the event becomes
//      internal to graph 2 and is used for a fork-join there. In both cases
//      the uses already recorded into graph 1 are preserved, and the new uses
//      are not captured by graph 1.
//   3. Signaling the event on a queue recording graph 1 again makes it
//      internal to graph 1 for a second fork-join.

#include "../../graph_common.hpp"

#include <sycl/ext/oneapi/experimental/reusable_events.hpp>
#include <sycl/properties/all_properties.hpp>

#include <optional>

void run_test(bool UseMultipleGraphs) {
  device Dev;
  context Ctx{Dev};

  queue Graph1Queue1{Ctx, Dev, {property::queue::in_order{}}};
  queue Graph1Queue2{Ctx, Dev, {property::queue::in_order{}}};
  // Records graph 2 when UseMultipleGraphs, otherwise executes eagerly.
  queue OtherQueue1{Ctx, Dev, {property::queue::in_order{}}};
  // Only created when UseMultipleGraphs, to fork-join within graph 2.
  std::optional<queue> OtherQueue2;
  if (UseMultipleGraphs)
    OtherQueue2.emplace(Ctx, Dev, property_list{property::queue::in_order{}});

  QueueStateVerifier verifier(Graph1Queue1, Graph1Queue2, OtherQueue1);
  auto VerifyOtherQueue2 = [&](exp_ext::queue_state Expected) {
    if (OtherQueue2)
      QueueStateVerifier{*OtherQueue2}.verify(Expected);
  };

  const size_t N = 1024;
  int *Graph1Data = malloc_device<int>(N, Dev, Ctx);
  int *OtherData = malloc_shared<int>(N, Dev, Ctx);
  std::iota(OtherData, OtherData + N, 0);

  exp_ext::command_graph Graph1{
      Ctx, Dev, {exp_ext::property::graph::enable_native_recording{}}};
  exp_ext::command_graph Graph2{
      Ctx, Dev, {exp_ext::property::graph::enable_native_recording{}}};

  event Ev = exp_ext::make_event(Ctx);

  Graph1.begin_recording(Graph1Queue1);
  if (UseMultipleGraphs) {
    Graph2.begin_recording(OtherQueue1);
    verifier.verify(RECORDING, EXECUTING, RECORDING);
    VerifyOtherQueue2(EXECUTING);
  } else {
    verifier.verify(RECORDING, EXECUTING, EXECUTING);
  }

  // Phase 1: Ev is internal to graph 1 and forks Graph1Queue1 to Graph1Queue2
  // and joins it back.
  Graph1Queue1.parallel_for(
      range<1>{N}, [=](id<1> Idx) { Graph1Data[Idx] = static_cast<int>(Idx); });
  exp_ext::enqueue_signal_event(Graph1Queue1, Ev);
  exp_ext::enqueue_wait_event(Graph1Queue2, Ev);
  Graph1Queue2.parallel_for(range<1>{N},
                            [=](id<1> Idx) { Graph1Data[Idx] *= 2; });
  exp_ext::enqueue_signal_event(Graph1Queue2, Ev);
  exp_ext::enqueue_wait_event(Graph1Queue1, Ev);
  Graph1Queue1.parallel_for(range<1>{N},
                            [=](id<1> Idx) { Graph1Data[Idx] += 1; });

  // Phase 2: Ev is signaled outside of graph 1, which moves it out of graph 1.
  if (UseMultipleGraphs) {
    // Ev becomes internal to graph 2 and forks OtherQueue1 to OtherQueue2 and
    // joins it back.
    OtherQueue1.parallel_for(range<1>{N},
                             [=](id<1> Idx) { OtherData[Idx] += 1; });
    exp_ext::enqueue_signal_event(OtherQueue1, Ev);
    exp_ext::enqueue_wait_event(*OtherQueue2, Ev);
    OtherQueue2->parallel_for(range<1>{N},
                              [=](id<1> Idx) { OtherData[Idx] *= 5; });
    exp_ext::enqueue_signal_event(*OtherQueue2, Ev);
    exp_ext::enqueue_wait_event(OtherQueue1, Ev);
    OtherQueue1.parallel_for(range<1>{N},
                             [=](id<1> Idx) { OtherData[Idx] += 4; });
    verifier.verify(RECORDING, RECORDING, RECORDING);
    VerifyOtherQueue2(RECORDING);
  } else {
    // Ev becomes eager. Waiting on it from the host must observe the eager
    // work that preceded the signal.
    OtherQueue1.parallel_for(range<1>{N},
                             [=](id<1> Idx) { OtherData[Idx] += 1; });
    OtherQueue1.parallel_for(range<1>{N},
                             [=](id<1> Idx) { OtherData[Idx] *= 5; });
    exp_ext::enqueue_signal_event(OtherQueue1, Ev);
    Ev.wait();
    assert(Ev.get_info<info::event::command_execution_status>() ==
           info::event_command_status::complete);
    for (size_t i = 0; i < N; i++)
      assert(check_value(i, (static_cast<int>(i) + 1) * 5, OtherData[i],
                         "OtherData after eager signal"));
    OtherQueue1.parallel_for(range<1>{N},
                             [=](id<1> Idx) { OtherData[Idx] += 4; });
    verifier.verify(RECORDING, RECORDING, EXECUTING);
  }

  // Phase 3: Ev is signaled on graph 1 again, which makes it internal to
  // graph 1 for a second fork-join.
  exp_ext::enqueue_signal_event(Graph1Queue1, Ev);
  exp_ext::enqueue_wait_event(Graph1Queue2, Ev);
  Graph1Queue2.parallel_for(range<1>{N},
                            [=](id<1> Idx) { Graph1Data[Idx] *= 3; });
  exp_ext::enqueue_signal_event(Graph1Queue2, Ev);
  exp_ext::enqueue_wait_event(Graph1Queue1, Ev);
  Graph1Queue1.parallel_for(range<1>{N},
                            [=](id<1> Idx) { Graph1Data[Idx] -= 7; });

  Graph1.end_recording();
  if (UseMultipleGraphs)
    Graph2.end_recording();
  verifier.verify(EXECUTING, EXECUTING, EXECUTING);
  VerifyOtherQueue2(EXECUTING);

  auto ExecGraph1 = Graph1.finalize();

  // Graph 1 initializes its own output, so replaying it is idempotent. Graph
  // 2 is replayed once so OtherData sees phase 2 exactly once on both paths.
  Graph1Queue1.ext_oneapi_graph(ExecGraph1);
  Graph1Queue1.ext_oneapi_graph(ExecGraph1);
  if (UseMultipleGraphs) {
    auto ExecGraph2 = Graph2.finalize();
    OtherQueue1.ext_oneapi_graph(ExecGraph2);
  }
  Graph1Queue1.wait_and_throw();
  OtherQueue1.wait_and_throw();

  std::vector<int> HostGraph1Data(N);
  Graph1Queue1.memcpy(HostGraph1Data.data(), Graph1Data, N * sizeof(int))
      .wait();

  // Graph 1 contains only the phase 1 and phase 3 fork-joins. Had graph 1
  // captured phase 2, its replays would apply phase 2 to OtherData again.
  for (size_t i = 0; i < N; i++) {
    int Graph1Expected = (static_cast<int>(i) * 2 + 1) * 3 - 7;
    int OtherExpected = (static_cast<int>(i) + 1) * 5 + 4;
    assert(check_value(i, Graph1Expected, HostGraph1Data[i], "Graph1Data"));
    assert(check_value(i, OtherExpected, OtherData[i], "OtherData"));
  }

  free(Graph1Data, Ctx);
  free(OtherData, Ctx);
}

int main() {
  run_test(/*UseMultipleGraphs=*/false);
  run_test(/*UseMultipleGraphs=*/true);
  return 0;
}
