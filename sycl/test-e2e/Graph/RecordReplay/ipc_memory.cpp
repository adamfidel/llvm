// REQUIRES: aspect-usm_device_allocations && aspect-ext_oneapi_ipc_memory
// UNSUPPORTED: windows
// UNSUPPORTED-INTENDED: This test relies on POSIX semantics (std::system +
//   binary self-exec) and level-zero IPC is unsupported on Windows..

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// Tests that IPC-shared USM device memory can be read and written by graphs
// in both the exporting and importing process, using the command-buffer path.

#include "../Inputs/ipc_memory.cpp"
