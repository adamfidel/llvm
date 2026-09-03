//==------------------------- NativeRecording.cpp --------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "NativeRecordingMock.hpp"

#include <gmock/gmock.h>

using NativeRecordingMock::failAfterWith;
using NativeRecordingMock::failBeforeWith;
using NativeRecordingMock::state;
using NativeRecordingMock::traceCount;
using NativeRecordingMock::traceIndex;
using ::testing::HasSubstr;

// Test that native recording throws when UR does not support it
TEST_F(NativeRecordingTest, NativeRecordingUnsupportedDevice) {
  state().SupportsNativeRecording = false;
  try {
    makeGraph();
    FAIL() << "Expected an exception";
  } catch (sycl::exception &E) {
    EXPECT_EQ(E.code(), sycl::errc::invalid);
  }
}

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

// The tests below check that a failure reported by a UR native recording entry
// point reaches the user with the originating UR error code preserved on the
// exception and named in what(). Graph operations that go through the graph
// implementation also name the operation they came from; the kernel enqueue,
// event wait and queue wait paths are shared with non-graph submissions and
// only report the generic UR error, so the UR code is what identifies the graph
// problem.

namespace {

// Runs Operation, which is expected to throw, and returns the message of the
// exception after checking the SYCL error code and the UR error code carried by
// it. Returns an empty message if nothing was thrown.
template <typename FnT>
std::string failureMessage(FnT Operation, sycl::errc ExpectedCode,
                           ur_result_t ExpectedUrError) {
  try {
    Operation();
  } catch (sycl::exception &E) {
    EXPECT_EQ(E.code(), ExpectedCode);
    EXPECT_EQ(sycl::detail::get_ur_error(E),
              static_cast<int32_t>(ExpectedUrError));
    return E.what();
  }
  ADD_FAILURE() << "Expected an exception";
  return {};
}

} // namespace

// A recording session that still has open forks is only detected when the
// capture is closed, so the error surfaces from end_recording. UR stops
// capturing before rejecting such a capture, hence the after-stage failure: the
// queue is left executing and there is nothing for the test to end afterwards.
TEST_F(NativeRecordingTest, UnjoinedForkDescriptiveError) {
  auto Graph = makeGraph();
  Graph.begin_recording(Queue);

  failAfterWith("urQueueEndGraphCaptureExp",
                UR_RESULT_ERROR_GRAPH_UNJOINED_FORKS);
  std::string Message =
      failureMessage([&]() { Graph.end_recording(Queue); }, sycl::errc::runtime,
                     UR_RESULT_ERROR_GRAPH_UNJOINED_FORKS);
  EXPECT_THAT(Message, HasSubstr("ending native graph capture"));
  EXPECT_THAT(Message, HasSubstr("UR_RESULT_ERROR_GRAPH_UNJOINED_FORKS"));

  EXPECT_EQ(Queue.ext_oneapi_get_state(), experimental::queue_state::executing);
}

// Using a graph-internal event outside of the graph is reported by the command
// that consumes it, i.e. a kernel enqueue during recording.
TEST_F(NativeRecordingTest, InternalEventDescriptiveError) {
  auto Graph = makeGraph();
  Graph.begin_recording(Queue);

  failBeforeWith("urEnqueueKernelLaunchWithArgsExp",
                 UR_RESULT_ERROR_GRAPH_INTERNAL_EVENT);
  std::string Message = failureMessage(
      [&]() {
        Queue.submit(
            [&](sycl::handler &CGH) { CGH.single_task<TestKernel>([]() {}); });
      },
      sycl::errc::runtime, UR_RESULT_ERROR_GRAPH_INTERNAL_EVENT);
  EXPECT_THAT(Message, HasSubstr("UR_RESULT_ERROR_GRAPH_INTERNAL_EVENT"));

  Graph.end_recording(Queue);
}

// A submission that would splice two recording sessions together is rejected by
// the kernel enqueue.
TEST_F(NativeRecordingTest, MergeAttemptDescriptiveError) {
  auto Graph = makeGraph();
  Graph.begin_recording(Queue);

  failBeforeWith("urEnqueueKernelLaunchWithArgsExp",
                 UR_RESULT_ERROR_GRAPH_CAPTURE_MERGE_ATTEMPT);
  std::string Message = failureMessage(
      [&]() {
        Queue.submit(
            [&](sycl::handler &CGH) { CGH.single_task<TestKernel>([]() {}); });
      },
      sycl::errc::runtime, UR_RESULT_ERROR_GRAPH_CAPTURE_MERGE_ATTEMPT);
  EXPECT_THAT(Message,
              HasSubstr("UR_RESULT_ERROR_GRAPH_CAPTURE_MERGE_ATTEMPT"));

  Graph.end_recording(Queue);
}

// An event signaled inside the recording belongs to the graph, so a host wait
// on it is unsupported while the capture is open.
TEST_F(NativeRecordingTest, RecordedEventHostWaitDescriptiveError) {
  auto Graph = makeGraph();
  Graph.begin_recording(Queue);

  sycl::event RecordedEvent = Queue.submit(
      [&](sycl::handler &CGH) { CGH.single_task<TestKernel>([]() {}); });

  failBeforeWith("urEventWait", UR_RESULT_ERROR_GRAPH_CAPTURE_UNSUPPORTED);
  std::string Message =
      failureMessage([&]() { RecordedEvent.wait(); }, sycl::errc::runtime,
                     UR_RESULT_ERROR_GRAPH_CAPTURE_UNSUPPORTED);
  EXPECT_THAT(Message, HasSubstr("UR_RESULT_ERROR_GRAPH_CAPTURE_UNSUPPORTED"));

  Graph.end_recording(Queue);
}

// Draining a queue that is recording would require executing the captured work,
// which is unsupported.
TEST_F(NativeRecordingTest, RecordingQueueWaitDescriptiveError) {
  auto Graph = makeGraph();
  Graph.begin_recording(Queue);

  failBeforeWith("urQueueFinish", UR_RESULT_ERROR_GRAPH_CAPTURE_UNSUPPORTED);
  std::string Message =
      failureMessage([&]() { Queue.wait(); }, sycl::errc::runtime,
                     UR_RESULT_ERROR_GRAPH_CAPTURE_UNSUPPORTED);
  EXPECT_THAT(Message, HasSubstr("UR_RESULT_ERROR_GRAPH_CAPTURE_UNSUPPORTED"));

  Graph.end_recording(Queue);
}

// Graph construction failures name graph creation, since the property that
// requested native recording is the only clue the user has at that point.
TEST_F(NativeRecordingTest, GraphCreateDescriptiveError) {
  failBeforeWith("urGraphCreateExp", UR_RESULT_ERROR_OUT_OF_RESOURCES);
  std::string Message =
      failureMessage([&]() { makeGraph(); }, sycl::errc::runtime,
                     UR_RESULT_ERROR_OUT_OF_RESOURCES);
  EXPECT_THAT(Message, HasSubstr("create native UR graph"));
  EXPECT_THAT(Message, HasSubstr("UR_RESULT_ERROR_OUT_OF_RESOURCES"));
}

// finalize() instantiates the recorded graph, so a graph rejected at that point
// is reported against the instantiation rather than the recording.
TEST_F(NativeRecordingTest, FinalizeDescriptiveError) {
  auto Graph = makeGraph();

  Graph.begin_recording(Queue);
  Queue.submit(
      [&](sycl::handler &CGH) { CGH.single_task<TestKernel>([]() {}); });
  Graph.end_recording(Queue);

  failBeforeWith("urGraphInstantiateGraphExp", UR_RESULT_ERROR_INVALID_GRAPH);
  std::string Message =
      failureMessage([&]() { Graph.finalize(); }, sycl::errc::runtime,
                     UR_RESULT_ERROR_INVALID_GRAPH);
  EXPECT_THAT(Message, HasSubstr("instantiate native UR executable graph"));
  EXPECT_THAT(Message, HasSubstr("UR_RESULT_ERROR_INVALID_GRAPH"));
}
