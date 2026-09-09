// REQUIRES: level_zero_v2_adapter && arch-intel_gpu_bmg_g21

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if level_zero %{%{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}

// Test that one event can be the external output of one native recording graph
// and the external input of another, ordering the two graphs against each other
// without any host synchronization between the two launches.
//
// The producing graph records a host to device copy rather than a kernel: BMG has
// a single compute engine, so two kernels are serialized by the hardware whether
// or not they are ordered by an event, whereas the recorded copy runs on a copy
// engine and can genuinely overlap with the consuming graph's kernel.
//
// Native recording is per context, so the two graphs are recorded sequentially.
//
// This cannot pass until the external signal and wait UR entry points exist;
// today both are no-ops (see the "[TRACE] EXTERNAL_*" prints in queue_impl.cpp).

#include "../../graph_common.hpp"

#include <sycl/ext/oneapi/experimental/reusable_events.hpp>
#include <sycl/properties/all_properties.hpp>

int main() {
  device Dev;
  context Ctx{Dev};

  queue Queue1{Ctx, Dev, {property::queue::in_order{}}};
  queue Queue2{Ctx, Dev, {property::queue::in_order{}}};

  QueueStateVerifier verifier(Queue1, Queue2);

  exp_ext::properties External{exp_ext::graph_external{}};
  event ExtEvent = exp_ext::make_event(Ctx);

  const size_t N = 1024;
  int *Data = malloc_device<int>(N, Dev, Ctx);

  std::vector<int> HostInput(N);
  std::iota(HostInput.begin(), HostInput.end(), 0);

  exp_ext::command_graph Graph1{
      Ctx, Dev, {exp_ext::property::graph::enable_native_recording{}}};
  exp_ext::command_graph Graph2{
      Ctx, Dev, {exp_ext::property::graph::enable_native_recording{}}};

  verifier.verify(EXECUTING, EXECUTING);

  // Graph 1 uploads the data and signals the event as its output.
  Graph1.begin_recording(Queue1);
  verifier.verify(RECORDING, EXECUTING);

  Queue1.memcpy(Data, HostInput.data(), N * sizeof(int));
  exp_ext::enqueue_signal_event(Queue1, ExtEvent, External);

  Graph1.end_recording();

  // Graph 2 waits on the same event as its input and consumes the data.
  Graph2.begin_recording(Queue2);
  verifier.verify(EXECUTING, RECORDING);

  exp_ext::enqueue_wait_event(Queue2, ExtEvent, External);
  Queue2.parallel_for(range<1>{N}, [=](id<1> Idx) { Data[Idx] *= 2; });

  Graph2.end_recording();
  verifier.verify(EXECUTING, EXECUTING);

  auto ExecutableGraph1 = Graph1.finalize();
  auto ExecutableGraph2 = Graph2.finalize();

  std::vector<int> HostData(N);
  for (unsigned Iter = 0; Iter < Iterations; Iter++) {
    Queue1.fill(Data, 0, N).wait();

    Queue1.ext_oneapi_graph(ExecutableGraph1);
    event GraphEvent2 = Queue2.submit(
        [&](handler &CGH) { CGH.ext_oneapi_graph(ExecutableGraph2); });
    GraphEvent2.wait();

    Queue2.memcpy(HostData.data(), Data, N * sizeof(int)).wait();
    for (size_t i = 0; i < N; i++) {
      assert(check_value(i, static_cast<int>(i) * 2, HostData[i], "HostData"));
    }
  }

  free(Data, Ctx);

  return 0;
}
