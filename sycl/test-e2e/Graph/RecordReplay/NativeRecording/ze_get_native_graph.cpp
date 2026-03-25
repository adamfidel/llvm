// REQUIRES: level_zero_v2_adapter, level_zero_dev_kit

// RUN: %{build} %level_zero_options -o %t.out
// RUN: %{run} %t.out
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if level_zero %{%{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}

// Tests get_native() for command graphs with native recording enabled

#include "../../graph_common.hpp"
#include "../../ze_common.hpp"
#include <limits>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <sycl/properties/all_properties.hpp>

int main() {
  queue Queue{property::queue::in_order{}};
  exp_ext::command_graph Graph{
      Queue.get_context(),
      Queue.get_device(),
      {exp_ext::property::graph::enable_native_recording{}}};

  const size_t N = 64;
  int *Data = malloc_device<int>(N, Queue);

  Graph.begin_recording(Queue);

  ze_graph_handle_t ModifiableHandle =
      get_native<backend::ext_oneapi_level_zero>(Graph);
  assert(ModifiableHandle != nullptr);

  ze_driver_handle_t ZeDriver = nullptr;
  zeGraphIsEmptyExp_fn GraphIsEmpty = nullptr;

  ASSERT_ZE_RESULT_SUCCESS(getDriver(ZeDriver));
  ASSERT_ZE_RESULT_SUCCESS(
      loadZeExtensionFunction(ZeDriver, "zeGraphIsEmptyExp", GraphIsEmpty));

  // Check that the handle points to the graph by asserting it is empty before
  // recording and non-empty after recording
  assert(ZE_RESULT_QUERY_TRUE == GraphIsEmpty(ModifiableHandle));

  Queue.submit([&](handler &CGH) {
    CGH.parallel_for(range<1>{N},
                     [=](id<1> idx) { Data[idx] = static_cast<int>(idx); });
  });

  assert(ZE_RESULT_QUERY_FALSE == GraphIsEmpty(ModifiableHandle));

  Graph.end_recording(Queue);

  auto ExecGraph = Graph.finalize();

  // Test get_native() returned correct executable graph by submitting and
  // checking the results after synchronization.
  ze_executable_graph_handle_t ExecutableHandle =
      get_native<backend::ext_oneapi_level_zero>(ExecGraph);
  assert(ExecutableHandle != nullptr);

  // With V2 adapter we will always get an immediate command list.
  ze_command_list_handle_t ZeCmdList = std::get<ze_command_list_handle_t>(
      get_native<backend::ext_oneapi_level_zero>(Queue));
  assert(ZeCmdList != nullptr);

  zeCommandListAppendGraphExp_fn AppendGraph = nullptr;
  ASSERT_ZE_RESULT_SUCCESS(loadZeExtensionFunction(
      ZeDriver, "zeCommandListAppendGraphExp", AppendGraph));
  ASSERT_ZE_RESULT_SUCCESS(
      AppendGraph(ZeCmdList, ExecutableHandle, nullptr, nullptr, 0, nullptr));
  ASSERT_ZE_RESULT_SUCCESS(zeCommandListHostSynchronize(
      ZeCmdList, std::numeric_limits<uint32_t>::max()));

  // Verify results
  std::vector<int> HostData(N);
  Queue.memcpy(HostData.data(), Data, N * sizeof(int)).wait();

  for (size_t i = 0; i < N; i++) {
    int Expected = static_cast<int>(i);
    assert(check_value(i, Expected, HostData[i], "HostData"));
  }
  free(Data, Queue);

  return 0;
}
