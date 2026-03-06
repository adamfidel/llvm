// REQUIRES: level_zero_v2_adapter, level_zero_dev_kit

// RUN: %{build} %level_zero_options -o %t.out
// RUN: env SYCL_GRAPH_ENABLE_NATIVE_RECORDING=1 %{run} %t.out
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if level_zero %{env SYCL_GRAPH_ENABLE_NATIVE_RECORDING=1 %{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}

// CHECK: Modifiable graph handle obtained successfully
// CHECK: Executable graph handle obtained successfully
// CHECK: Queue native handle obtained successfully
// CHECK: Test passed

// Tests get_native() for command graphs with native recording enabled

#include "../../graph_common.hpp"
#include "../../ze_graph_common.hpp"
// SYCL
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <sycl/properties/all_properties.hpp>

int main() {
  // Initialize Level Zero driver
  ze_result_t result = zeInit(ZE_INIT_FLAG_GPU_ONLY);
  if (result != ZE_RESULT_SUCCESS) {
    std::cout << "zeInit failed with error code: " << result << std::endl;
    return 1;
  }

  // Create queue with immediate command list property for native recording
  queue Queue{{property::queue::in_order{},
               ext::intel::property::queue::immediate_command_list{}}};

  // Create graph - native recording enabled via
  // SYCL_GRAPH_ENABLE_NATIVE_RECORDING
  exp_ext::command_graph Graph{Queue.get_context(), Queue.get_device()};

  const size_t N = 64;
  int *Data = malloc_device<int>(N, Queue);

  // Use queue recording mode to create the graph
  Graph.begin_recording(Queue);

  // Record a simple kernel that sets Data[i] = i
  Queue.submit([&](handler &CGH) {
    CGH.parallel_for(range<1>{N},
                     [=](id<1> idx) { Data[idx] = static_cast<int>(idx); });
  });

  Graph.end_recording(Queue);

  // Test get_native() for modifiable graph
  ze_graph_handle_t ModifiableHandle;
  try {
    ModifiableHandle = get_native<backend::ext_oneapi_level_zero>(Graph);
    std::cout << "Modifiable graph handle obtained successfully" << std::endl;
  } catch (const sycl::exception &e) {
    std::cout << "Error getting modifiable graph handle: " << e.what()
              << std::endl;
    free(Data, Queue);
    return 1;
  }

  // Verify handle is non-null
  if (ModifiableHandle == nullptr) {
    std::cout << "Error: modifiable graph handle is null" << std::endl;
    free(Data, Queue);
    return 1;
  }

  // Finalize to executable graph
  auto ExecGraph = Graph.finalize();

  // Test get_native() for executable graph
  ze_executable_graph_handle_t ExecutableHandle;
  try {
    ExecutableHandle = get_native<backend::ext_oneapi_level_zero>(ExecGraph);
    std::cout << "Executable graph handle obtained successfully" << std::endl;
  } catch (const sycl::exception &e) {
    std::cout << "Error getting executable graph handle: " << e.what()
              << std::endl;
    free(Data, Queue);
    return 1;
  }

  // Verify handle is non-null
  if (ExecutableHandle == nullptr) {
    std::cout << "Error: executable graph handle is null" << std::endl;
    free(Data, Queue);
    return 1;
  }

  // Verify we can get the native queue handle as well
  auto ZeQueue = get_native<backend::ext_oneapi_level_zero>(Queue);
  ze_command_list_handle_t ZeCmdList = nullptr;

  // The queue returns a variant, extract the command list
  if (std::holds_alternative<ze_command_list_handle_t>(ZeQueue)) {
    ZeCmdList = std::get<ze_command_list_handle_t>(ZeQueue);
    if (ZeCmdList == nullptr) {
      std::cout << "Error: Queue command list handle is null" << std::endl;
      free(Data, Queue);
      return 1;
    }
    std::cout << "Queue native handle obtained successfully" << std::endl;
  } else {
    std::cout << "Error: Queue does not have immediate command list"
              << std::endl;
    free(Data, Queue);
    return 1;
  }

  // Execute the graph using SYCL API and verify correctness
  Queue.submit([&](handler &CGH) { CGH.ext_oneapi_graph(ExecGraph); });
  Queue.wait();

  // Verify results
  std::vector<int> HostData(N);
  Queue.memcpy(HostData.data(), Data, N * sizeof(int)).wait();

  bool passed = true;
  for (size_t i = 0; i < N; i++) {
    int Expected = static_cast<int>(i);
    if (!check_value(i, Expected, HostData[i], "HostData")) {
      passed = false;
      break;
    }
  }

  free(Data, Queue);

  if (passed) {
    std::cout << "Test passed" << std::endl;
    return 0;
  } else {
    std::cout << "Test failed: incorrect results" << std::endl;
    return 1;
  }
}
