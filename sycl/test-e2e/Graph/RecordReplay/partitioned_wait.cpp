// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if level_zero %{%{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}

// Tests partitioned wait feature in SYCL Graph.
// This test demonstrates how queue.wait() calls during recording create 
// dummy nodes that partition the graph into "before", barrier, and "after" subgraphs.

#include "../graph_common.hpp"

#include <sycl/properties/all_properties.hpp>

int main() {
  property_list Properties{property::queue::in_order{}};
  queue Queue{Properties};

  exp_ext::command_graph Graph{Queue.get_context(), Queue.get_device()};

  const size_t N = 100;
  int *A = malloc_device<int>(N, Queue);
  int *B = malloc_device<int>(N, Queue);
  int *C = malloc_device<int>(N, Queue);
  int *D = malloc_device<int>(N, Queue);

  // Initialize data outside recording
  Queue.submit([&](handler &CGH) {
    CGH.parallel_for(N, [=](id<1> it) {
      A[it] = static_cast<int>(it);
      B[it] = 0;
      C[it] = 0;
      D[it] = 0;
    });
  }).wait();

  // Begin recording the graph
  Graph.begin_recording(Queue);

  // Part 1: "Before" subgraph operations
  auto Event1 = Queue.submit([&](handler &CGH) {
    CGH.parallel_for(N, [=](id<1> it) {
      B[it] = A[it] * 2; // B = A * 2
    });
  });

  auto Event2 = Queue.submit([&](handler &CGH) {
    CGH.depends_on(Event1);
    CGH.parallel_for(N, [=](id<1> it) {
      C[it] = B[it] + 1; // C = B + 1 = A * 2 + 1
    });
  });

  // This queue.wait() should create a dummy barrier node in the graph
  // instead of asserting, partitioning the graph into before/after sections
  Queue.wait();

  // Part 2: "After" subgraph operations 
  // These operations should be in the "after" partition
  auto Event3 = Queue.submit([&](handler &CGH) {
    CGH.parallel_for(N, [=](id<1> it) {
      D[it] = C[it] * 3; // D = C * 3 = (A * 2 + 1) * 3
    });
  });

  // Another wait should create another barrier
  Queue.wait();

  // More operations after second barrier
  Queue.submit([&](handler &CGH) {
    CGH.parallel_for(N, [=](id<1> it) {
      D[it] = D[it] + A[it]; // D = D + A = (A * 2 + 1) * 3 + A = 6*A + 3 + A = 7*A + 3
    });
  });

  Graph.end_recording();

  // Finalize the graph - this should trigger partitioning
  auto ExecGraph = Graph.finalize();

  // Execute the partitioned graph
  Queue.submit([&](handler &CGH) { CGH.ext_oneapi_graph(ExecGraph); });
  Queue.wait_and_throw();

  // Verify results
  std::vector<int> OutputA(N), OutputB(N), OutputC(N), OutputD(N);
  Queue.memcpy(OutputA.data(), A, N * sizeof(int)).wait();
  Queue.memcpy(OutputB.data(), B, N * sizeof(int)).wait();
  Queue.memcpy(OutputC.data(), C, N * sizeof(int)).wait();
  Queue.memcpy(OutputD.data(), D, N * sizeof(int)).wait();

  // Check results
  for (size_t i = 0; i < N; i++) {
    int expected_a = static_cast<int>(i);
    int expected_b = expected_a * 2;
    int expected_c = expected_b + 1;
    int expected_d = 7 * expected_a + 3;

    assert(check_value(i, expected_a, OutputA[i], "A"));
    assert(check_value(i, expected_b, OutputB[i], "B"));
    assert(check_value(i, expected_c, OutputC[i], "C"));
    assert(check_value(i, expected_d, OutputD[i], "D"));
  }

  // Test multiple execution of the partitioned graph
  // Reset data
  Queue.submit([&](handler &CGH) {
    CGH.parallel_for(N, [=](id<1> it) {
      A[it] = static_cast<int>(i) + 10; // Different input
      B[it] = 0;
      C[it] = 0;
      D[it] = 0;
    });
  }).wait();

  // Execute again
  Queue.submit([&](handler &CGH) { CGH.ext_oneapi_graph(ExecGraph); });
  Queue.wait_and_throw();

  // Verify second execution
  Queue.memcpy(OutputA.data(), A, N * sizeof(int)).wait();
  Queue.memcpy(OutputB.data(), B, N * sizeof(int)).wait();
  Queue.memcpy(OutputC.data(), C, N * sizeof(int)).wait();
  Queue.memcpy(OutputD.data(), D, N * sizeof(int)).wait();

  for (size_t i = 0; i < N; i++) {
    int expected_a = static_cast<int>(i) + 10;
    int expected_b = expected_a * 2;
    int expected_c = expected_b + 1;
    int expected_d = 7 * expected_a + 3;

    assert(check_value(i, expected_a, OutputA[i], "A (second execution)"));
    assert(check_value(i, expected_b, OutputB[i], "B (second execution)"));
    assert(check_value(i, expected_c, OutputC[i], "C (second execution)"));
    assert(check_value(i, expected_d, OutputD[i], "D (second execution)"));
  }

  // Clean up
  sycl::free(A, Queue);
  sycl::free(B, Queue);
  sycl::free(C, Queue);
  sycl::free(D, Queue);

  return 0;
}
