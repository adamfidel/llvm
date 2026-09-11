//==------------------------- NativeRecording.cpp --------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "NativeRecordingMock.hpp"

#include <detail/context_impl.hpp>

using NativeRecordingMock::expectFailure;
using NativeRecordingMock::state;
using NativeRecordingMock::traceCount;
using NativeRecordingMock::traceIndex;

// Traces UR recording layer
TEST_F(NativeRecordingTest, RecordingUrTrace) {
  auto Graph = makeGraph();

  Graph.begin_recording(Queue);
  Queue.submit(
      [&](sycl::handler &CGH) { CGH.single_task<TestKernel>([]() {}); });
  Graph.end_recording(Queue);

  ASSERT_EQ(traceCount("urQueueBeginCaptureIntoGraphExp"), 1u);
  ASSERT_EQ(traceCount("urEnqueueKernelLaunchWithArgsExp"), 1u);
  ASSERT_EQ(traceCount("urQueueEndGraphCaptureExp"), 1u);
  EXPECT_LT(traceIndex("urQueueBeginCaptureIntoGraphExp"),
            traceIndex("urEnqueueKernelLaunchWithArgsExp"));
  EXPECT_LT(traceIndex("urEnqueueKernelLaunchWithArgsExp"),
            traceIndex("urQueueEndGraphCaptureExp"));
}

// Finalize and submission traces
TEST_F(NativeRecordingTest, FinalizeSubmitUrTrace) {
  auto Graph = makeGraph();

  Graph.begin_recording(Queue);
  Queue.submit(
      [&](sycl::handler &CGH) { CGH.single_task<TestKernel>([]() {}); });
  Graph.end_recording(Queue);

  EXPECT_EQ(traceCount("urGraphInstantiateGraphExp"), 0u);

  auto ExecGraph = Graph.finalize();

  EXPECT_EQ(traceCount("urGraphInstantiateGraphExp", nativeHandle(Graph)), 1u);
  ASSERT_NE(nativeHandle(ExecGraph), nullptr);
  EXPECT_EQ(traceCount("urEnqueueGraphExp"), 0u);

  Queue.ext_oneapi_graph(ExecGraph);
  Queue.wait();

  EXPECT_EQ(traceCount("urEnqueueGraphExp", nativeHandle(ExecGraph)), 1u);
  EXPECT_EQ(traceCount("urCommandBufferCreateExp"), 0u);
}

// The executable graph must be destroyed prior to the modifiable.
TEST_F(NativeRecordingTest, DestructionOrder) {
  ur_exp_graph_handle_t GraphHandle = nullptr;
  ur_exp_executable_graph_handle_t ExecHandle = nullptr;
  {
    auto ExecGraph = [&]() {
      auto Graph = makeGraph();
      GraphHandle = nativeHandle(Graph);

      Graph.begin_recording(Queue);
      Queue.submit(
          [&](sycl::handler &CGH) { CGH.single_task<TestKernel>([]() {}); });
      Graph.end_recording(Queue);

      return Graph.finalize();
    }();
    ExecHandle = nativeHandle(ExecGraph);

    ASSERT_NE(GraphHandle, nullptr);
    ASSERT_NE(ExecHandle, nullptr);
    EXPECT_EQ(traceCount("urGraphDestroyExp"), 0u);
    EXPECT_EQ(traceCount("urGraphExecutableGraphDestroyExp"), 0u);
  }

  EXPECT_EQ(traceCount("urGraphExecutableGraphDestroyExp", ExecHandle), 1u);
  EXPECT_EQ(traceCount("urGraphDestroyExp", GraphHandle), 1u);
  EXPECT_LT(traceIndex("urGraphExecutableGraphDestroyExp"),
            traceIndex("urGraphDestroyExp"));
}

// Check that destruction callback goes through UR and not SYCL command buffer
// path.
TEST_F(NativeRecordingTest, DestructionCallbackUrTrace) {
  bool CallbackFired1 = false;
  bool CallbackFired2 = false;
  ur_exp_graph_handle_t Handle = nullptr;
  {
    auto Graph = makeGraph();
    Handle = nativeHandle(Graph);

    EXPECT_EQ(traceCount("urGraphCreateExp", Handle), 1u);
    ASSERT_NE(Handle, nullptr);

    Graph.set_destruction_callback(
        [&CallbackFired1]() { CallbackFired1 = true; });
    Graph.set_destruction_callback(
        [&CallbackFired2]() { CallbackFired2 = true; });

    EXPECT_EQ(traceCount("urGraphSetDestructionCallbackExp", Handle), 2u);
    EXPECT_FALSE(CallbackFired1);
    EXPECT_FALSE(CallbackFired2);
    EXPECT_EQ(traceCount("urGraphDestroyExp"), 0u);
  }

  EXPECT_EQ(traceCount("urGraphDestroyExp", Handle), 1u);
  EXPECT_LT(traceIndex("urGraphCreateExp"), traceIndex("urGraphDestroyExp"));
  EXPECT_LT(traceIndex("urGraphSetDestructionCallbackExp"),
            traceIndex("urGraphDestroyExp"));
  EXPECT_TRUE(CallbackFired1);
  EXPECT_TRUE(CallbackFired2);
}

// Check that the graph ID is going through UR and not the SYCL command buffer
// or native recording fallback path.
TEST_F(NativeRecordingTest, GetIdUrTrace) {
  auto Graph = makeGraph();
  EXPECT_EQ(Graph.get_id(), NativeRecordingMock::FirstGraphId);
  EXPECT_EQ(traceCount("urGraphGetIdExp", nativeHandle(Graph)), 1u);
}

// Check UR call for get graph and graph uniqueness
TEST_F(NativeRecordingTest, GetGraphUrTrace) {
  auto Graph = makeGraph();
  auto SecondGraph = makeGraph();
  sycl::queue SecondQueue{Dev, {sycl::property::queue::in_order{}}};

  Graph.begin_recording(Queue);
  SecondGraph.begin_recording(SecondQueue);

  auto RecordedGraph = Queue.ext_oneapi_get_graph();
  auto SecondRecordedGraph = SecondQueue.ext_oneapi_get_graph();

  EXPECT_EQ(traceCount("urQueueGetGraphExp"), 2u);
  EXPECT_EQ(getSyclObjImpl(RecordedGraph), getSyclObjImpl(Graph));
  EXPECT_EQ(getSyclObjImpl(SecondRecordedGraph), getSyclObjImpl(SecondGraph));
  EXPECT_EQ(nativeHandle(RecordedGraph), nativeHandle(Graph));
  EXPECT_EQ(nativeHandle(SecondRecordedGraph), nativeHandle(SecondGraph));

  Graph.end_recording(Queue);
  SecondGraph.end_recording(SecondQueue);
}

// Check UR empty graph call
TEST_F(NativeRecordingTest, EmptyUrTrace) {
  auto Graph = makeGraph();
  ur_exp_graph_handle_t Handle = nativeHandle(Graph);

  state().graph(Handle).IsEmpty = true;
  EXPECT_TRUE(Graph.empty());
  EXPECT_EQ(traceCount("urGraphIsEmptyExp", Handle), 1u);

  state().graph(Handle).IsEmpty = false;
  EXPECT_FALSE(Graph.empty());
  EXPECT_EQ(traceCount("urGraphIsEmptyExp", Handle), 2u);
}

// Check UR call for queue state
TEST_F(NativeRecordingTest, GetStateUrTrace) {
  auto Graph = makeGraph();
  EXPECT_EQ(Queue.ext_oneapi_get_state(), experimental::queue_state::executing);

  Graph.begin_recording(Queue);
  EXPECT_EQ(Queue.ext_oneapi_get_state(), experimental::queue_state::recording);

  Graph.end_recording(Queue);
  EXPECT_EQ(Queue.ext_oneapi_get_state(), experimental::queue_state::executing);

  EXPECT_GE(traceCount("urQueueIsGraphCaptureEnabledExp"), 3u);
}

// Events are flagged as potentially recorded to provide better error messaging
// on invalid cross capture dependencies
TEST_F(NativeRecordingTest, PotentiallyNativeRecordedEvents) {
  sycl::queue ExecutingQueue{
      Queue.get_context(), Dev, {sycl::property::queue::in_order{}}};
  int HostVal = 42;
  int *DevPtr = sycl::malloc_device<int>(1, Queue);
  auto Graph = makeGraph();

  auto submitOps = [&](sycl::queue &Q) {
    return std::vector<std::pair<std::string, sycl::event>>{
        {"barrier handler",
         Q.submit([&](sycl::handler &CGH) { CGH.ext_oneapi_barrier(); })},
        {"barrier shortcut", Q.ext_oneapi_submit_barrier()},
        {"fill handler",
         Q.submit([&](sycl::handler &CGH) { CGH.fill(DevPtr, 0, 1); })},
        {"fill shortcut", Q.fill(DevPtr, 0, 1)},
        {"memset handler", Q.submit([&](sycl::handler &CGH) {
           CGH.memset(DevPtr, 0, sizeof(int));
         })},
        {"memset shortcut", Q.memset(DevPtr, 0, sizeof(int))},
        {"memcpy handler", Q.submit([&](sycl::handler &CGH) {
           CGH.memcpy(DevPtr, &HostVal, sizeof(int));
         })},
        {"memcpy shortcut", Q.memcpy(DevPtr, &HostVal, sizeof(int))},
        {"kernel handler", Q.submit([&](sycl::handler &CGH) {
           CGH.single_task<TestKernel>([]() {});
         })},
        {"kernel shortcut", Q.single_task<TestKernel>([]() {})}};
  };

  auto expectRecorded = [&](sycl::queue &Q, bool Expected) {
    for (const auto &[Name, Event] : submitOps(Q))
      EXPECT_EQ(getSyclObjImpl(Event)->isPotentiallyNativeRecorded(), Expected)
          << Name;
  };

  expectRecorded(Queue, false);

  Graph.begin_recording(Queue);
  expectRecorded(Queue, true);
  // The flag is per context and assumes a direct native submission could
  // transition the recording, so all queue submissions are this point are
  // potentially native recorded.
  expectRecorded(ExecutingQueue, true);

  Graph.end_recording(Queue);
  expectRecorded(Queue, false);

  sycl::free(DevPtr, Queue);
}

// Recordings from queues in the same context are counted, so the context stays
// active until the last recording ends. The counter walks 0 -> 1 -> 2 -> 1 -> 0
// here, of which isNativeRecordingActive() exposes "non-zero".
TEST_F(NativeRecordingTest, ContextRecordingActive) {
  sycl::queue SecondQueue{Dev, {sycl::property::queue::in_order{}}};
  sycl::detail::context_impl &Ctx = *getSyclObjImpl(Queue.get_context());
  ASSERT_EQ(SecondQueue.get_context(), Queue.get_context());

  auto Graph = makeGraph();
  auto SecondGraph = makeGraph();
  EXPECT_FALSE(Ctx.isNativeRecordingActive());

  Graph.begin_recording(Queue);
  EXPECT_TRUE(Ctx.isNativeRecordingActive());

  SecondGraph.begin_recording(SecondQueue);
  EXPECT_TRUE(Ctx.isNativeRecordingActive());

  Graph.end_recording(Queue);
  EXPECT_TRUE(Ctx.isNativeRecordingActive());

  SecondGraph.end_recording(SecondQueue);
  EXPECT_FALSE(Ctx.isNativeRecordingActive());
}

// A recording left open when the graph is destroyed still ends on the context.
TEST_F(NativeRecordingTest, ContextRecordingActiveGraphDestroyed) {
  sycl::detail::context_impl &Ctx = *getSyclObjImpl(Queue.get_context());
  {
    auto Graph = makeGraph();
    Graph.begin_recording(Queue);
    EXPECT_TRUE(Ctx.isNativeRecordingActive());
  }
  EXPECT_FALSE(Ctx.isNativeRecordingActive());
}

// Destroying the recording queue does not end the recording, the graph does.
TEST_F(NativeRecordingTest, ContextRecordingActiveQueueDestroyed) {
  sycl::detail::context_impl &Ctx = *getSyclObjImpl(Queue.get_context());
  {
    auto Graph = makeGraph();
    {
      sycl::queue RecordingQueue{Dev, {sycl::property::queue::in_order{}}};
      ASSERT_EQ(RecordingQueue.get_context(), Queue.get_context());
      Graph.begin_recording(RecordingQueue);
      EXPECT_TRUE(Ctx.isNativeRecordingActive());
    }
    EXPECT_TRUE(Ctx.isNativeRecordingActive());
  }
  EXPECT_FALSE(Ctx.isNativeRecordingActive());
}

// Recording a graph without native support enabled leaves the context inactive.
TEST_F(NativeRecordingTest, ContextRecordingActiveNonNativeGraph) {
  sycl::context SyclCtx = Queue.get_context();
  sycl::detail::context_impl &Ctx = *getSyclObjImpl(SyclCtx);
  ModifiableGraph Graph{SyclCtx, Dev};

  Graph.begin_recording(Queue);
  EXPECT_FALSE(Ctx.isNativeRecordingActive());

  Graph.end_recording(Queue);
  EXPECT_FALSE(Ctx.isNativeRecordingActive());
  EXPECT_EQ(traceCount("urQueueBeginCaptureIntoGraphExp"), 0u);
}

// A failed begin capture must leave the counter at zero.
TEST_F(NativeRecordingTest, ContextRecordingActiveBeginFailure) {
  sycl::detail::context_impl &Ctx = *getSyclObjImpl(Queue.get_context());
  auto Graph = makeGraph();

  FAIL_UR_BEFORE(urQueueBeginCaptureIntoGraphExp,
                 UR_RESULT_ERROR_GRAPH_CAPTURE_UNSUPPORTED);
  expectFailure([&]() { Graph.begin_recording(Queue); },
                UR_RESULT_ERROR_GRAPH_CAPTURE_UNSUPPORTED);
  EXPECT_FALSE(Ctx.isNativeRecordingActive());
}

// End capture reporting a failure after the queue already left capture mode
// still takes the counter back to zero.
TEST_F(NativeRecordingTest, ContextRecordingActiveEndFailure) {
  sycl::detail::context_impl &Ctx = *getSyclObjImpl(Queue.get_context());
  auto Graph = makeGraph();

  Graph.begin_recording(Queue);
  ASSERT_TRUE(Ctx.isNativeRecordingActive());

  FAIL_UR_AFTER(urQueueEndGraphCaptureExp, UR_RESULT_ERROR_INVALID_QUEUE);
  expectFailure([&]() { Graph.end_recording(Queue); },
                UR_RESULT_ERROR_INVALID_QUEUE);
  EXPECT_FALSE(Ctx.isNativeRecordingActive());
  EXPECT_EQ(Queue.ext_oneapi_get_state(), experimental::queue_state::executing);
}
