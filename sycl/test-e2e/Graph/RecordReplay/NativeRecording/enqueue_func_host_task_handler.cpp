// REQUIRES: level_zero_v2_adapter && arch-intel_gpu_bmg_g21
// REQUIRES-INTEL-DRIVER: lin: 37561, win: 101.8724

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// RUN: %if level_zero %{%{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}

// Tests fork/join of a restricted host_task (submit_with_event + handler path)
// across two in-order queues in a native-recording SYCL Graph. Exercises the
// host task consuming a dependency event and producing a signal event.

#define GRAPH_E2E_NATIVE_RECORDING

#include "../Inputs/enqueue_func_host_task_handler.cpp"
