// REQUIRES: level_zero_v2_adapter && arch-intel_gpu_bmg_g21

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if level_zero %{%{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}

// Test the per-replay contract of an external wait recorded into a native
// recording graph: every replay needs its own signal, and each replay consumes
// exactly one. The recorded kernel increments the buffer by an operand which is
// only uploaded after the graph has been launched, so a replay running ahead of
// its signal increments by zero. The buffer is never reset, so the value after
// replay k has to be exactly k.
//
// The operand is uploaded with a host to device copy: BMG has a single compute
// engine, so two kernels are serialized by the hardware whether or not they are
// ordered by an event, whereas a copy runs on a copy engine and can genuinely
// overlap with the graph's kernel.
//
// The event is never signaled before it is recorded, which is the case that goes
// through event_impl::materializeExternalEvent.
//
// This cannot pass until the external wait UR entry point exists; today the
// recorded wait is a no-op (see "[TRACE] EXTERNAL_WAIT" in queue_impl.cpp).

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
  int *Operand = malloc_device<int>(1, Dev, Ctx);
  Queue1.fill(Data, 0, N).wait();

  const int HostOperand = 1;

  exp_ext::command_graph Graph{
      Ctx, Dev, {exp_ext::property::graph::enable_native_recording{}}};

  verifier.verify(EXECUTING, EXECUTING);

  Graph.begin_recording(Queue1);
  verifier.verify(RECORDING, EXECUTING);

  exp_ext::enqueue_wait_event(Queue1, ExtEvent, External);
  Queue1.parallel_for(range<1>{N},
                      [=](id<1> Idx) { Data[Idx] += Operand[0]; });

  Graph.end_recording();
  verifier.verify(EXECUTING, EXECUTING);

  auto ExecutableGraph = Graph.finalize();

  // TODO: it is unspecified whether signaling twice before a single replay is a
  // counting or a binary semantic, so this test only signals once per replay.
  std::vector<int> HostData(N);
  for (unsigned Iter = 1; Iter <= Iterations; Iter++) {
    Queue2.fill(Operand, 0, 1).wait();

    event GraphEvent = Queue1.submit(
        [&](handler &CGH) { CGH.ext_oneapi_graph(ExecutableGraph); });

    Queue2.memcpy(Operand, &HostOperand, sizeof(int));
    exp_ext::enqueue_signal_event(Queue2, ExtEvent);

    GraphEvent.wait();

    Queue2.memcpy(HostData.data(), Data, N * sizeof(int)).wait();
    for (size_t i = 0; i < N; i++) {
      assert(check_value(i, static_cast<int>(Iter), HostData[i], "HostData"));
    }
  }

  free(Data, Ctx);
  free(Operand, Ctx);

  return 0;
}
