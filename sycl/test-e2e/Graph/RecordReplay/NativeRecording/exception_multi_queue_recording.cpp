// REQUIRES: level_zero_v2_adapter && arch-intel_gpu_bmg_g21

// RUN: %{build} -o %t.out
// RUN: env SYCL_GRAPH_ENABLE_NATIVE_RECORDING=1 %{run} %t.out
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if level_zero %{env SYCL_GRAPH_ENABLE_NATIVE_RECORDING=1 %{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}

#include "../../graph_common.hpp"

#include <sycl/properties/all_properties.hpp>

int main() {
  device Dev;
  context Ctx{Dev};

  // Create two in-order queues sharing the same device and context
  queue Queue1{Ctx,
               Dev,
               {property::queue::in_order{},
                ext::intel::property::queue::immediate_command_list{}}};
  queue Queue2{Ctx,
               Dev,
               {property::queue::in_order{},
                ext::intel::property::queue::immediate_command_list{}}};

  exp_ext::command_graph Graph{Ctx, Dev};

  constexpr size_t N = 1024;
  int *Data = malloc_device<int>(N, Dev, Ctx);

  // Start recording on Queue1
  Graph.begin_recording(Queue1);

  // Submit a kernel to Queue1
  Queue1.submit([&](handler &CGH) {
    CGH.parallel_for(range<1>{N}, [=](id<1> idx) { Data[idx] = idx; });
  });

  // Try to start recording on Queue2 while Queue1 is still recording
  if (!expectException([&]() { Graph.begin_recording(Queue2); },
                       "begin_recording on second queue")) {
    Graph.end_recording(Queue1);
    free(Data, Ctx);
    return 1;
  }

  // End recording on Queue1
  Graph.end_recording(Queue1);

  // Verify that after ending recording on Queue1, we CAN start recording on
  // Queue2 (sequential recording to different queues should work)
  bool sequentialRecordingWorks = true;
  try {
    Graph.begin_recording(Queue2);
    Queue2.submit([&](handler &CGH) {
      CGH.parallel_for(range<1>{N}, [=](id<1> idx) { Data[idx] = idx * 2; });
    });
    Graph.end_recording(Queue2);
  } catch (const sycl::exception &e) {
    std::cerr << "ERROR: Sequential recording to different queues should work!"
              << std::endl;
    sequentialRecordingWorks = false;
  }

  if (!sequentialRecordingWorks) {
    free(Data, Ctx);
    return 1;
  }

  free(Data, Ctx);
  return 0;
}
