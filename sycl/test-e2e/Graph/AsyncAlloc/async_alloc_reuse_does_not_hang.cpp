// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if level_zero %{%{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}

// Regression test for PR #21170
// Tests that allocation reuse in graphs does not cause an infinite loop.
// The bug was in tryReuseExistingAllocation() where if a node had already been
// visited (MTotalVisitedEdges > 0), the code would continue without popping
// the node from NodesToCheck queue, causing an infinite loop when checking if
// an allocation can be reused.

#include "../graph_common.hpp"
#include <sycl/ext/oneapi/experimental/async_alloc/async_alloc.hpp>

using T = int;

int main() {
  queue Queue{};

  exp_ext::command_graph Graph{Queue.get_context(), Queue.get_device()};

  std::vector<T> OutputData1(Size);
  std::vector<T> OutputData2(Size);
  std::vector<T> ReferenceData(Size);

  std::iota(ReferenceData.begin(), ReferenceData.end(), 0);

  // Create a graph with two separate allocations that will be freed and
  // could potentially be reused. The bug would manifest when checking if
  // an allocation can be reused and a node was visited multiple times.

  // First allocation and kernel
  T *AsyncPtr1 = nullptr;
  auto AllocNode1 = Graph.add([&](handler &CGH) {
    AsyncPtr1 = static_cast<T *>(
        exp_ext::async_malloc(CGH, usm::alloc::device, Size * sizeof(T)));
  });

  auto KernelNode1 = Graph.add(
      [&](handler &CGH) {
        CGH.parallel_for(range<1>(Size), [=](item<1> ID) {
          size_t LinID = ID.get_linear_id();
          AsyncPtr1[LinID] = static_cast<T>(LinID);
        });
      },
      {exp_ext::property::node::depends_on{AllocNode1}});

  auto CopyNode1 = Graph.add(
      [&](handler &CGH) {
        CGH.memcpy(OutputData1.data(), AsyncPtr1, Size * sizeof(T));
      },
      {exp_ext::property::node::depends_on{KernelNode1}});

  auto FreeNode1 = Graph.add(
      [&](handler &CGH) { exp_ext::async_free(CGH, AsyncPtr1); },
      {exp_ext::property::node::depends_on{CopyNode1}});

  // Second allocation and kernel that depends on first free
  // This creates a scenario where the allocation reuse logic needs to
  // traverse through previously visited nodes
  T *AsyncPtr2 = nullptr;
  auto AllocNode2 = Graph.add(
      [&](handler &CGH) {
        AsyncPtr2 = static_cast<T *>(
            exp_ext::async_malloc(CGH, usm::alloc::device, Size * sizeof(T)));
      },
      {exp_ext::property::node::depends_on{FreeNode1}});

  auto KernelNode2 = Graph.add(
      [&](handler &CGH) {
        CGH.parallel_for(range<1>(Size), [=](item<1> ID) {
          size_t LinID = ID.get_linear_id();
          AsyncPtr2[LinID] = static_cast<T>(LinID) * 2;
        });
      },
      {exp_ext::property::node::depends_on{AllocNode2}});

  auto CopyNode2 = Graph.add(
      [&](handler &CGH) {
        CGH.memcpy(OutputData2.data(), AsyncPtr2, Size * sizeof(T));
      },
      {exp_ext::property::node::depends_on{KernelNode2}});

  Graph.add([&](handler &CGH) { exp_ext::async_free(CGH, AsyncPtr2); },
            {exp_ext::property::node::depends_on{CopyNode2}});

  // Before the fix, finalizing this graph would hang in tryReuseExistingAllocation
  // because nodes would be checked multiple times without being removed from
  // the NodesToCheck queue
  auto GraphExec = Graph.finalize();

  Queue.ext_oneapi_graph(GraphExec).wait_and_throw();

  // Verify results
  for (size_t i = 0; i < Size; i++) {
    assert(check_value(i, ReferenceData[i], OutputData1[i], "OutputData1"));
    assert(check_value(i, ReferenceData[i] * 2, OutputData2[i], "OutputData2"));
  }

  std::cout << "Test passed!" << std::endl;

  return 0;
}
