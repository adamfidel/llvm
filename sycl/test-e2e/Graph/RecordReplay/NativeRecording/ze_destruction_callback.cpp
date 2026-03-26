// REQUIRES: level_zero_v2_adapter, level_zero_dev_kit
// RUN: %{build} %level_zero_options -o %t.out
// RUN: %{run} %t.out | FileCheck %s
// RUN: %if level_zero %{%{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}

// Tests zeGraphSetDestructionCallbackExp for native recording graphs

#include "../../graph_common.hpp"
#include "../../ze_common.hpp"
#include <level_zero/ze_api.h>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <sycl/properties/all_properties.hpp>

struct CallbackData {
  void *allocated_memory;
  sycl::context *sycl_context;
};

void printCallback(void *pUserData) {
  std::cout << "CALLBACK_INVOKED" << std::endl;
}

void memoryCleanupCallback(void *pUserData) {
  CallbackData *data = static_cast<CallbackData *>(pUserData);
  if (data->allocated_memory && data->sycl_context) {
    sycl::free(data->allocated_memory, *(data->sycl_context));
    data->allocated_memory = nullptr;
  }
}

int main() {
  queue Queue{property::queue::in_order{}};
  auto Context = Queue.get_context();
  auto Device = Queue.get_device();

  exp_ext::command_graph Graph{
      Context, Device, {exp_ext::property::graph::enable_native_recording{}}};

  const size_t N = 64;
  int *Data = malloc_device<int>(N, Queue);
  void *ExtraMemory = malloc_device(1024 * 1024, Queue);

  ze_driver_handle_t ZeDriver = nullptr;
  ASSERT_ZE_RESULT_SUCCESS(getDriver(ZeDriver));

  Graph.begin_recording(Queue);
  ze_graph_handle_t ModifiableHandle =
      get_native<backend::ext_oneapi_level_zero>(Graph);
  assert(ModifiableHandle != nullptr);

  zeGraphSetDestructionCallbackExp_fn SetDestructionCallback = nullptr;
  ASSERT_ZE_RESULT_SUCCESS(loadZeExtensionFunction(
      ZeDriver, "zeGraphSetDestructionCallbackExp", SetDestructionCallback));

  CallbackData cbData = {ExtraMemory, &Context};
  ASSERT_ZE_RESULT_SUCCESS(SetDestructionCallback(
      ModifiableHandle, printCallback, nullptr, nullptr));
  ASSERT_ZE_RESULT_SUCCESS(SetDestructionCallback(
      ModifiableHandle, memoryCleanupCallback, &cbData, nullptr));

  Queue.submit([&](handler &CGH) {
    CGH.parallel_for(range<1>{N},
                     [=](id<1> idx) { Data[idx] = static_cast<int>(idx); });
  });

  Graph.end_recording(Queue);
  auto ExecGraph = Graph.finalize();

  Queue.submit([&](handler &CGH) { CGH.ext_oneapi_graph(ExecGraph); });
  Queue.wait();

  std::vector<int> HostData(N);
  Queue.memcpy(HostData.data(), Data, N * sizeof(int)).wait();
  for (size_t i = 0; i < N; i++) {
    assert(HostData[i] == static_cast<int>(i));
  }

  free(Data, Queue);

  std::cout << "BEFORE_GRAPH_DESTRUCTION" << std::endl;
  // CHECK: BEFORE_GRAPH_DESTRUCTION

  {
    auto temp = std::move(ExecGraph);
  }
  {
    auto temp = std::move(Graph);
  }
  // CHECK: CALLBACK_INVOKED

  std::cout << "AFTER_GRAPH_DESTRUCTION" << std::endl;
  // CHECK: AFTER_GRAPH_DESTRUCTION

  assert(cbData.allocated_memory == nullptr && "Memory should have been freed");

  return 0;
}
