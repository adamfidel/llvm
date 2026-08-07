// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if level_zero %{%{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}

// Regression test for a self-deadlock when transitive queue recording is
// triggered by ext_oneapi_set_external_event instead of by handler::depends_on
// inside a command-group function. The external event comes from an in-order
// queue that a graph is recording, so the next submission to the second
// in-order queue must transition that queue into recording mode on the same
// graph and record a node depending on the external event's node. Before the
// fix that submission never returned.
//
// Reported as https://github.com/intel/llvm/issues/20563.

#include "../graph_common.hpp"
#include <sycl/properties/all_properties.hpp>

int main() {
  using T = int;

  device Dev;
  context Ctx{Dev};

  property_list InOrderProp = {property::queue::in_order{}};
  queue Q1{Ctx, Dev, InOrderProp};
  queue Q2{Ctx, Dev, InOrderProp};

  std::vector<T> DataA(Size), DataB(Size);
  std::iota(DataA.begin(), DataA.end(), 1);
  std::iota(DataB.begin(), DataB.end(), 100);
  std::vector<T> ReferenceA(DataA), ReferenceB(DataB);

  T *PtrA = malloc_device<T>(Size, Q1);
  T *PtrB = malloc_device<T>(Size, Q1);
  Q1.copy(DataA.data(), PtrA, Size);
  Q1.copy(DataB.data(), PtrB, Size);
  Q1.wait_and_throw();

  exp_ext::command_graph Graph{Ctx, Dev};
  Graph.begin_recording(Q1);
  assert(Q1.ext_oneapi_get_state() == RECORDING);
  assert(Q2.ext_oneapi_get_state() == EXECUTING);

  auto GraphEventA = Q1.submit([&](handler &CGH) {
    CGH.parallel_for(range<1>(Size), [=](item<1> Id) { PtrA[Id]++; });
  });

  // Equivalent to a depends_on(GraphEventA) in the next submission to Q2, so
  // it must pull Q2 into recording mode on Graph.
  Q2.ext_oneapi_set_external_event(GraphEventA);

  auto GraphEventB = Q2.submit([&](handler &CGH) {
    CGH.parallel_for(range<1>(Size), [=](item<1> Id) { PtrB[Id] *= 2; });
  });

  assert(Q2.ext_oneapi_get_state() == RECORDING);
  assert(Q2.ext_oneapi_get_graph() == Graph);

  Graph.end_recording();
  assert(Q1.ext_oneapi_get_state() == EXECUTING);
  assert(Q2.ext_oneapi_get_state() == EXECUTING);

  // Neither recorded command may have executed yet.
  Q1.copy(PtrA, DataA.data(), Size);
  Q1.copy(PtrB, DataB.data(), Size);
  Q1.wait_and_throw();
  for (size_t i = 0; i < Size; i++) {
    assert(check_value(i, ReferenceA[i], DataA[i], "DataA"));
    assert(check_value(i, ReferenceB[i], DataB[i], "DataB"));
  }

  auto GraphExec = Graph.finalize();
  Q1.ext_oneapi_graph(GraphExec);
  Q1.wait_and_throw();

  Q1.copy(PtrA, DataA.data(), Size);
  Q1.copy(PtrB, DataB.data(), Size);
  Q1.wait_and_throw();
  for (size_t i = 0; i < Size; i++) {
    assert(check_value(i, ReferenceA[i] + 1, DataA[i], "DataA"));
    assert(check_value(i, ReferenceB[i] * 2, DataB[i], "DataB"));
  }

  free(PtrA, Q1);
  free(PtrB, Q1);
  return 0;
}
