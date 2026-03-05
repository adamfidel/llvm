// REQUIRES: level_zero_v2_adapter && arch-intel_gpu_bmg_g21

// RUN: %{build} -o %t.out
// RUN: env SYCL_GRAPH_ENABLE_NATIVE_RECORDING=1 %{run} %t.out
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if level_zero %{env SYCL_GRAPH_ENABLE_NATIVE_RECORDING=1 %{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}

// Simplified fork-join test to expose cmdlist index bug.
// Queue1 (recorded, cmdlist 0) forks to Queue2 (not recorded, cmdlist != 0).
// Queue1 must join back by waiting on Queue2's event.

#include "../../graph_common.hpp"
#include <sycl/properties/all_properties.hpp>

int main() {
  queue Queue1{};
  queue Queue2{Queue1.get_context(), Queue1.get_device()};

  int *DataA = malloc_device<int>(1, Queue1);
  int *DataB = malloc_device<int>(1, Queue2);
  int *DataC = malloc_device<int>(1, Queue2);

  Queue1.fill(DataA, 0, 1).wait();
  Queue2.fill(DataB, 0, 1).wait();
  Queue2.fill(DataC, 0, 1).wait();

  // Start recording Queue1 (uses cmdlist 0)
  exp_ext::command_graph Graph{Queue1.get_context(), Queue1.get_device()};
  Graph.begin_recording(Queue1);

  // Queue1 work (recorded, cmdlist 0)
  auto EventQ1 = Queue1.single_task([=]() { DataA[0] = 42; });

  // Fork: Queue2 depends on Queue1's recorded event (cmdlist 0 -> cmdlist X)
  // BUG: If Queue2 uses cmdlist != 0, cross-cmdlist event dependency may fail
  auto EventQ2 = Queue2.submit([&](handler &CGH) {
    CGH.depends_on(EventQ1);
    CGH.single_task([=]() { DataB[0] = DataA[0] * 2; });
  });

  auto EventQ3 = Queue2.submit([&](handler &CGH) {
    // CGH.depends_on(EventQ2);
    CGH.single_task([=]() { DataC[0] = 96; });
  });

  // Join: Queue1 waits on Queue2's event (cmdlist X -> cmdlist 0)
  Queue1.ext_oneapi_submit_barrier({EventQ2});

  Graph.end_recording();
  Queue2.wait();
  int HostC = 0;
  Queue2.memcpy(&HostC, DataC, sizeof(int)).wait();
  std::cout << HostC << std::endl;
  assert(HostC == 0 && "Queue2 result submitted eagerly");

  // Execute graph
  auto GraphExec = Graph.finalize();
  Queue1.submit([&](handler &CGH) { CGH.ext_oneapi_graph(GraphExec); }).wait();
  Queue2.wait();

  // Verify results
  int HostA = 0, HostB = 0;
  Queue1.memcpy(&HostA, DataA, sizeof(int)).wait();
  Queue2.memcpy(&HostB, DataB, sizeof(int)).wait();

  assert(HostA == 42 && "Queue1 result incorrect");
  assert(HostB == 84 && "Queue2 result incorrect (fork-join failed)");

  free(DataA, Queue1);
  free(DataB, Queue2);
  free(DataC, Queue2);

  std::cout << "Test passed" << std::endl;
  return 0;
}
