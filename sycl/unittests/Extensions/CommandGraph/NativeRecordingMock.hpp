//==--------------------- NativeRecordingMock.hpp---------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Mock UR native graph support and test fixture for the SYCL native-recording
// path.

#pragma once

#include "Common.hpp"

#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace NativeRecordingMock {

// UR function name and associated object handle for tracing UR calls.
struct TraceEntry {
  std::string EntryPoint;
  const void *Handle;
};

// Unique per-graph state
struct GraphState {
  uint64_t Id = 0;
  bool IsEmpty = true;
  std::vector<ur_exp_graph_destruction_callback_t> DestructionCallbacks = {};
  std::vector<void *> DestructionCallbacksData = {};
};

inline constexpr uint64_t FirstGraphId = 1;

// The failure a test asked an entry point to report. Mock callbacks are plain
// function pointers and cannot capture, so the entry point and the code live
// here rather than in the installed callback. At most one failure is live at a
// time, which is all a test needs and keeps the callbacks free of a lookup.
struct InjectedFailure {
  std::string EntryPoint;
  ur_result_t Error = UR_RESULT_SUCCESS;
};

// Singleton state the mock callbacks share.
struct MockState {
  bool SupportsNativeRecording = true;

  std::vector<TraceEntry> Trace;
  std::unordered_map<ur_exp_graph_handle_t, GraphState> Graphs;
  std::unordered_map<ur_queue_handle_t, ur_exp_graph_handle_t> PrimaryCapturing;
  uint64_t NextGraphId = FirstGraphId;
  InjectedFailure Failure;

  GraphState &graph(ur_exp_graph_handle_t Handle);
};

MockState &state();

// Fails EntryPoint before its mock implementation runs, modelling UR rejecting
// the call outright: the mock state is left untouched. The call is still
// traced.
void failBeforeWith(std::string EntryPoint, ur_result_t Error);

// Fails EntryPoint after its mock implementation ran, modelling UR doing the
// work and then reporting a problem: the trace and the mock state reflect a
// completed call, but the entry point returns Error. Use this where the mock
// state has to keep up with the failure, e.g. a capture that UR closed before
// rejecting it.
void failAfterWith(std::string EntryPoint, ur_result_t Error);

size_t traceCount(std::string_view EntryPoint);

// Calls to EntryPoint that were about Handle, for tests with more than one
// graph or queue in flight.
size_t traceCount(std::string_view EntryPoint, const void *Handle);

// Position of the first call to EntryPoint, so that comparing two positions
// orders two calls. Fails the test and returns npos if it was never called.
size_t traceIndex(std::string_view EntryPoint);

// Resets the mock state and registers the callbacks. Must run after the UrMock
// constructor. The default callbacks are meant to
// simulate "success" states. To test error handling, the user should override
// these defaults as needed.
void registerDefaultCallbacks();

} // namespace NativeRecordingMock

// Fixture for the native-recording path
class NativeRecordingTest : public ::testing::Test {
public:
  NativeRecordingTest();

protected:
  using ModifiableGraph =
      experimental::command_graph<experimental::graph_state::modifiable>;
  using ExecutableGraph =
      experimental::command_graph<experimental::graph_state::executable>;

  ModifiableGraph makeGraph();

  static ur_exp_graph_handle_t nativeHandle(const ModifiableGraph &Graph);

  static ur_exp_executable_graph_handle_t
  nativeHandle(const ExecutableGraph &ExecGraph);

  unittest::UrMock<> Mock;
  sycl::platform Plat;
  sycl::device Dev;
  sycl::queue Queue;
};
