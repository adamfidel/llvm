//==-------------------- ExternalEventRecording.cpp ------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Transitive queue recording triggered by ext_oneapi_set_external_event rather
// than by handler::depends_on inside a command-group function.
//
// Per the Transitive Queue Recording section of sycl_ext_oneapi_graph, an
// external event produced by a recording queue must pull the target queue into
// recording mode on the same graph before the next command-group is submitted,
// and that command-group must become a graph node depending on the external
// event's node.
//
// The failure mode these tests gate against is a hang inside the submission,
// not a wrong value, so every submission that could deadlock runs under an
// explicit deadline.

#include "Common.hpp"

#include <chrono>
#include <cstdlib>
#include <functional>
#include <future>
#include <iostream>
#include <thread>

using namespace sycl;
using namespace sycl::ext::oneapi;

namespace {

constexpr auto SubmissionDeadline = std::chrono::seconds(30);

// A deadlocked submission holds the queue mutex forever: the thread cannot be
// interrupted and cannot be joined, so the process has to end for the result to
// be observable at all. Terminating with a distinct status turns an indefinite
// hang into a reportable outcome; the deadline is never reached once the
// submission completes.
void runWithDeadline(const char *What, const std::function<void()> &Body) {
  std::packaged_task<void()> Task{Body};
  std::future<void> Done = Task.get_future();
  std::thread Worker{std::move(Task)};
  if (Done.wait_for(SubmissionDeadline) == std::future_status::timeout) {
    std::cout.flush();
    std::cerr << "\nDEADLOCK: " << What << " did not complete within "
              << SubmissionDeadline.count() << "s\n";
    std::cerr.flush();
    std::_Exit(66);
  }
  Worker.join();
  // Rethrows on this thread so callers can assert on the exception.
  Done.get();
}

int EagerKernelLaunches = 0;

ur_result_t countEagerKernelLaunch(void *) {
  ++EagerKernelLaunches;
  return UR_RESULT_SUCCESS;
}

void countEagerKernelLaunches() {
  EagerKernelLaunches = 0;
  mock::getCallbacks().set_after_callback("urEnqueueKernelLaunchWithArgsExp",
                                          &countEagerKernelLaunch);
}

void expectEdge(const experimental::node &Pred,
                const experimental::node &Succ) {
  auto Successors = Pred.get_successors();
  ASSERT_EQ(Successors.size(), 1lu);
  EXPECT_EQ(Successors[0], Succ);
  auto Predecessors = Succ.get_predecessors();
  ASSERT_EQ(Predecessors.size(), 1lu);
  EXPECT_EQ(Predecessors[0], Pred);
}

sycl::errc codeOfThrownException(const char *What,
                                 const std::function<void()> &Body) {
  try {
    runWithDeadline(What, Body);
  } catch (const sycl::exception &E) {
    return static_cast<sycl::errc>(E.code().value());
  }
  return sycl::errc::success;
}

} // namespace

// C2a, C3, C4: the target in-order queue has had no work submitted to it, so
// the submission takes the no-last-event finalizer path.
TEST_F(CommandGraphTest, ExternalEventTransitionEmptyQueue) {
  sycl::property_list InOrder{sycl::property::queue::in_order()};
  context Ctx{Dev};
  queue Q1{Ctx, Dev, InOrder};
  queue Q2{Ctx, Dev, InOrder};

  experimental::command_graph<experimental::graph_state::modifiable> Graph{Ctx,
                                                                           Dev};
  Graph.begin_recording(Q1);
  countEagerKernelLaunches();

  auto EventQ1 = Q1.submit(
      [&](sycl::handler &CGH) { CGH.single_task<TestKernel>([]() {}); });

  Q2.ext_oneapi_set_external_event(EventQ1);

  sycl::event EventQ2;
  runWithDeadline("submission to a queue holding a graph external event", [&] {
    EventQ2 = Q2.submit(
        [&](sycl::handler &CGH) { CGH.single_task<TestKernel>([]() {}); });
  });

  ASSERT_EQ(Q2.ext_oneapi_get_state(), experimental::queue_state::recording);
  EXPECT_EQ(Q2.ext_oneapi_get_graph(), Graph);

  // Recorded, not executed: neither submission may reach the device before the
  // graph is finalized and submitted.
  EXPECT_EQ(EagerKernelLaunches, 0);
  ASSERT_EQ(Graph.get_nodes().size(), 2lu);

  expectEdge(experimental::node::get_node_from_event(EventQ1),
             experimental::node::get_node_from_event(EventQ2));

  Graph.end_recording();
}

// C2b, C3, C4: the target in-order queue has prior non-graph work whose event
// is still pending, so the submission takes the with-deps finalizer path. The
// blocked host task is what makes that path deterministic.
TEST_F(CommandGraphTest, ExternalEventTransitionQueueWithPriorWork) {
  sycl::property_list InOrder{sycl::property::queue::in_order()};
  context Ctx{Dev};
  queue Q1{Ctx, Dev, InOrder};
  queue Q2{Ctx, Dev, InOrder};

  std::mutex HostTaskMutex;
  std::unique_lock<std::mutex> Lock{HostTaskMutex, std::defer_lock};
  Lock.lock();
  Q2.submit([&](sycl::handler &CGH) {
    CGH.host_task([&HostTaskMutex]() {
      std::lock_guard<std::mutex> Wait{HostTaskMutex};
    });
  });

  experimental::command_graph<experimental::graph_state::modifiable> Graph{Ctx,
                                                                           Dev};
  Graph.begin_recording(Q1);
  countEagerKernelLaunches();

  auto EventQ1 = Q1.submit(
      [&](sycl::handler &CGH) { CGH.single_task<TestKernel>([]() {}); });

  Q2.ext_oneapi_set_external_event(EventQ1);

  sycl::event EventQ2;
  runWithDeadline("submission to a primed queue holding a graph external event",
                  [&] {
                    EventQ2 = Q2.submit([&](sycl::handler &CGH) {
                      CGH.single_task<TestKernel>([]() {});
                    });
                  });

  ASSERT_EQ(Q2.ext_oneapi_get_state(), experimental::queue_state::recording);
  EXPECT_EQ(Q2.ext_oneapi_get_graph(), Graph);
  EXPECT_EQ(EagerKernelLaunches, 0);

  // The prior host task is not part of the graph.
  ASSERT_EQ(Graph.get_nodes().size(), 2lu);
  expectEdge(experimental::node::get_node_from_event(EventQ1),
             experimental::node::get_node_from_event(EventQ2));

  Graph.end_recording();
  Lock.unlock();
  Q2.wait();
}

// C2c, C3, C4: the submission on the target queue is a host task, so it takes
// the host-task finalizer path.
TEST_F(CommandGraphTest, ExternalEventTransitionHostTaskSubmission) {
  sycl::property_list InOrder{sycl::property::queue::in_order()};
  context Ctx{Dev};
  queue Q1{Ctx, Dev, InOrder};
  queue Q2{Ctx, Dev, InOrder};

  experimental::command_graph<experimental::graph_state::modifiable> Graph{Ctx,
                                                                           Dev};
  Graph.begin_recording(Q1);

  auto EventQ1 = Q1.submit(
      [&](sycl::handler &CGH) { CGH.single_task<TestKernel>([]() {}); });

  Q2.ext_oneapi_set_external_event(EventQ1);

  sycl::event EventQ2;
  runWithDeadline("host task submitted to a queue holding a graph external "
                  "event",
                  [&] {
                    EventQ2 = Q2.submit(
                        [&](sycl::handler &CGH) { CGH.host_task([]() {}); });
                  });

  ASSERT_EQ(Q2.ext_oneapi_get_state(), experimental::queue_state::recording);
  EXPECT_EQ(Q2.ext_oneapi_get_graph(), Graph);

  ASSERT_EQ(Graph.get_nodes().size(), 2lu);
  auto NodeQ2 = experimental::node::get_node_from_event(EventQ2);
  EXPECT_EQ(NodeQ2.get_type(), experimental::node_type::host_task);
  expectEdge(experimental::node::get_node_from_event(EventQ1), NodeQ2);

  Graph.end_recording();
  Q2.wait();
}

// C5: a queue that transitions to recording partway through a submission must
// keep its dependency bookkeeping coherent afterwards. Continuing to record on
// it, ending recording, and then submitting outside the graph must all behave,
// and the graph must hold exactly the three recorded commands.
TEST_F(CommandGraphTest, ExternalEventTransitionBookkeeping) {
  sycl::property_list InOrder{sycl::property::queue::in_order()};
  context Ctx{Dev};
  queue Q1{Ctx, Dev, InOrder};
  queue Q2{Ctx, Dev, InOrder};

  std::mutex HostTaskMutex;
  std::unique_lock<std::mutex> Lock{HostTaskMutex, std::defer_lock};
  Lock.lock();
  Q2.submit([&](sycl::handler &CGH) {
    CGH.host_task([&HostTaskMutex]() {
      std::lock_guard<std::mutex> Wait{HostTaskMutex};
    });
  });

  experimental::command_graph<experimental::graph_state::modifiable> Graph{Ctx,
                                                                           Dev};
  Graph.begin_recording(Q1);
  countEagerKernelLaunches();

  auto EventQ1 = Q1.submit(
      [&](sycl::handler &CGH) { CGH.single_task<TestKernel>([]() {}); });

  Q2.ext_oneapi_set_external_event(EventQ1);

  sycl::event EventTransition;
  runWithDeadline("submission that transitions the queue mid-flight", [&] {
    EventTransition = Q2.submit(
        [&](sycl::handler &CGH) { CGH.single_task<TestKernel>([]() {}); });
  });
  ASSERT_EQ(Q2.ext_oneapi_get_state(), experimental::queue_state::recording);

  sycl::event EventRecorded;
  runWithDeadline("further submission to the transitioned queue", [&] {
    EventRecorded = Q2.submit(
        [&](sycl::handler &CGH) { CGH.single_task<TestKernel>([]() {}); });
  });

  ASSERT_EQ(Graph.get_nodes().size(), 3lu);
  auto NodeQ1 = experimental::node::get_node_from_event(EventQ1);
  auto NodeTransition =
      experimental::node::get_node_from_event(EventTransition);
  auto NodeRecorded = experimental::node::get_node_from_event(EventRecorded);
  expectEdge(NodeQ1, NodeTransition);
  expectEdge(NodeTransition, NodeRecorded);
  EXPECT_EQ(EagerKernelLaunches, 0);

  Graph.end_recording();
  EXPECT_EQ(Q1.ext_oneapi_get_state(), experimental::queue_state::executing);
  EXPECT_EQ(Q2.ext_oneapi_get_state(), experimental::queue_state::executing);

  runWithDeadline("submission after recording ended", [&] {
    Q2.submit(
        [&](sycl::handler &CGH) { CGH.single_task<TestKernel>([]() {}); });
  });

  // The post-recording submission must leave the graph untouched and must
  // actually reach the device.
  EXPECT_EQ(Graph.get_nodes().size(), 3lu);
  Lock.unlock();
  Q2.wait();
  EXPECT_GE(EagerKernelLaunches, 1);
}

// C7: an external event the target queue may not depend on still produces
// errc::invalid rather than being silently dropped.
TEST_F(CommandGraphTest, ExternalEventCrossContextThrows) {
  sycl::property_list InOrder{sycl::property::queue::in_order()};
  context Ctx1{Dev};
  context Ctx2{Dev};
  queue Q1{Ctx1, Dev, InOrder};
  queue Q2{Ctx2, Dev, InOrder};

  experimental::command_graph<experimental::graph_state::modifiable> Graph{Ctx1,
                                                                           Dev};
  Graph.begin_recording(Q1);
  auto EventQ1 = Q1.submit(
      [&](sycl::handler &CGH) { CGH.single_task<TestKernel>([]() {}); });

  EXPECT_EQ(codeOfThrownException("cross-context external event",
                                  [&] {
                                    Q2.ext_oneapi_set_external_event(EventQ1);
                                    Q2.submit([&](sycl::handler &CGH) {
                                      CGH.single_task<TestKernel>([]() {});
                                    });
                                  }),
            sycl::errc::invalid);

  Graph.end_recording();
}

TEST_F(CommandGraphTest, ExternalEventCrossDeviceThrows) {
  auto Devices = device::get_devices();
  if (Devices.size() < 2)
    GTEST_SKIP();

  sycl::property_list InOrder{sycl::property::queue::in_order()};
  device &Dev1 = Devices[0];
  device &Dev2 = Devices[1];
  context Ctx{{Dev1, Dev2}};
  queue Q1{Ctx, Dev1, InOrder};
  queue Q2{Ctx, Dev2, InOrder};

  experimental::command_graph<experimental::graph_state::modifiable> Graph{
      Ctx, Dev1};
  Graph.begin_recording(Q1);
  auto EventQ1 = Q1.submit(
      [&](sycl::handler &CGH) { CGH.single_task<TestKernel>([]() {}); });

  EXPECT_EQ(codeOfThrownException("cross-device external event",
                                  [&] {
                                    Q2.ext_oneapi_set_external_event(EventQ1);
                                    Q2.submit([&](sycl::handler &CGH) {
                                      CGH.single_task<TestKernel>([]() {});
                                    });
                                  }),
            sycl::errc::invalid);

  Graph.end_recording();
}

TEST_F(CommandGraphTest, ExternalEventFromOtherGraphThrows) {
  sycl::property_list InOrder{sycl::property::queue::in_order()};
  context Ctx{Dev};
  queue Q1{Ctx, Dev, InOrder};
  queue Q2{Ctx, Dev, InOrder};

  experimental::command_graph<experimental::graph_state::modifiable> Graph1{
      Ctx, Dev};
  experimental::command_graph<experimental::graph_state::modifiable> Graph2{
      Ctx, Dev};
  Graph1.begin_recording(Q1);
  Graph2.begin_recording(Q2);

  auto EventQ1 = Q1.submit(
      [&](sycl::handler &CGH) { CGH.single_task<TestKernel>([]() {}); });

  EXPECT_EQ(codeOfThrownException("external event from a different graph",
                                  [&] {
                                    Q2.ext_oneapi_set_external_event(EventQ1);
                                    Q2.submit([&](sycl::handler &CGH) {
                                      CGH.single_task<TestKernel>([]() {});
                                    });
                                  }),
            sycl::errc::invalid);

  Graph1.end_recording();
  Graph2.end_recording();
}

TEST_F(CommandGraphTest, ExternalEventOnOutOfOrderQueueThrows) {
  sycl::property_list InOrder{sycl::property::queue::in_order()};
  context Ctx{Dev};
  queue Q1{Ctx, Dev, InOrder};
  queue OutOfOrderQueue{Ctx, Dev};

  experimental::command_graph<experimental::graph_state::modifiable> Graph{Ctx,
                                                                           Dev};
  Graph.begin_recording(Q1);
  auto EventQ1 = Q1.submit(
      [&](sycl::handler &CGH) { CGH.single_task<TestKernel>([]() {}); });

  EXPECT_EQ(
      codeOfThrownException(
          "ext_oneapi_set_external_event on an out-of-order queue",
          [&] { OutOfOrderQueue.ext_oneapi_set_external_event(EventQ1); }),
      sycl::errc::invalid);

  Graph.end_recording();
}
