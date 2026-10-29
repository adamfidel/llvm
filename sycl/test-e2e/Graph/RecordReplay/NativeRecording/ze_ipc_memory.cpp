// REQUIRES: level_zero_v2_adapter && level_zero_dev_kit && (arch-intel_gpu_bmg_g21 || arch-intel_gpu_bmg_g31 || arch-intel_gpu_cri)
// REQUIRES: aspect-usm_device_allocations && aspect-ext_oneapi_ipc_memory
// UNSUPPORTED: windows
// UNSUPPORTED-INTENDED: This test relies on POSIX semantics (std::system +
//   binary self-exec) and pidfd_getfd(2) to share the IPC file descriptor.
// UNSUPPORTED: ze_debug
// UNSUPPORTED-INTENDED: L0 adapter incorrectly reports memory leaks because it
// doesn't take into account direct calls to the L0 API.

// RUN: %{build} %level_zero_options -o %t.out
// RUN: %{run} %t.out

// Tests that USM device memory shared through the Level Zero IPC API can be
// read and written by native recording graphs in both the exporting and
// importing process.

#define GRAPH_E2E_NATIVE_RECORDING
#define USE_ZE_IPC

#include "../../Inputs/ipc_memory.cpp"
