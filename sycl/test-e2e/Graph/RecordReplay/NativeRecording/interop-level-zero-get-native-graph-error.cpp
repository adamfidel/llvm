// REQUIRES: level_zero_v2_adapter, level_zero_dev_kit

// RUN: %{build} %level_zero_options -o %t.out
// RUN: %{run} %t.out

// CHECK: Caught expected exception for modifiable graph
// CHECK: Error code: {{[0-9]+}}
// CHECK: Message: {{.*}}native recording{{.*}}
// CHECK: Caught expected exception for executable graph
// CHECK: Error code: {{[0-9]+}}
// CHECK: Message: {{.*}}native recording{{.*}}
// CHECK: Test passed: exceptions thrown as expected

// Tests that get_native() throws exception when native recording is disabled

#include "../../graph_common.hpp"
// Level-Zero
#include <level_zero/ze_api.h>
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

  // Create queue with immediate command list but WITHOUT native recording
  // enabled
  queue Queue{{property::queue::in_order{},
               ext::intel::property::queue::immediate_command_list{}}};

  // Create graph - native recording is NOT enabled (no environment variable)
  exp_ext::command_graph Graph{Queue.get_context(), Queue.get_device()};

  const size_t N = 64;
  int *Data = malloc_device<int>(N, Queue);

  // Use queue recording mode to create the graph
  Graph.begin_recording(Queue);

  // Record a simple kernel
  Queue.submit([&](handler &CGH) {
    CGH.parallel_for(range<1>{N},
                     [=](id<1> idx) { Data[idx] = static_cast<int>(idx); });
  });

  Graph.end_recording(Queue);

  // Test get_native() for modifiable graph - should throw
  bool caughtModifiableException = false;
  try {
    ze_graph_handle_t ModifiableHandle =
        get_native<backend::ext_oneapi_level_zero>(Graph);
    std::cout << "Error: get_native() should have thrown for modifiable graph"
              << std::endl;
    free(Data, Queue);
    return 1;
  } catch (const sycl::exception &e) {
    caughtModifiableException = true;
    std::cout << "Caught expected exception for modifiable graph" << std::endl;
    std::cout << "Error code: " << e.code().value() << std::endl;
    std::cout << "Message: " << e.what() << std::endl;

    // Verify error code is feature_not_supported
    if (e.code() != make_error_code(errc::feature_not_supported)) {
      std::cout << "Error: unexpected error code" << std::endl;
      free(Data, Queue);
      return 1;
    }

    // Verify message mentions native recording
    std::string message = e.what();
    if (message.find("native recording") == std::string::npos) {
      std::cout << "Error: exception message doesn't mention native recording"
                << std::endl;
      free(Data, Queue);
      return 1;
    }
  }

  if (!caughtModifiableException) {
    std::cout << "Error: did not catch exception for modifiable graph"
              << std::endl;
    free(Data, Queue);
    return 1;
  }

  // Finalize to executable graph
  auto ExecGraph = Graph.finalize();

  // Test get_native() for executable graph - should also throw
  bool caughtExecutableException = false;
  try {
    ze_executable_graph_handle_t ExecutableHandle =
        get_native<backend::ext_oneapi_level_zero>(ExecGraph);
    std::cout << "Error: get_native() should have thrown for executable graph"
              << std::endl;
    free(Data, Queue);
    return 1;
  } catch (const sycl::exception &e) {
    caughtExecutableException = true;
    std::cout << "Caught expected exception for executable graph" << std::endl;
    std::cout << "Error code: " << e.code().value() << std::endl;
    std::cout << "Message: " << e.what() << std::endl;

    // Verify error code is feature_not_supported
    if (e.code() != make_error_code(errc::feature_not_supported)) {
      std::cout << "Error: unexpected error code" << std::endl;
      free(Data, Queue);
      return 1;
    }

    // Verify message mentions native recording
    std::string message = e.what();
    if (message.find("native recording") == std::string::npos) {
      std::cout << "Error: exception message doesn't mention native recording"
                << std::endl;
      free(Data, Queue);
      return 1;
    }
  }

  if (!caughtExecutableException) {
    std::cout << "Error: did not catch exception for executable graph"
              << std::endl;
    free(Data, Queue);
    return 1;
  }

  free(Data, Queue);

  std::cout << "Test passed: exceptions thrown as expected" << std::endl;
  return 0;
}
