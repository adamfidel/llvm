// REQUIRES: level_zero_v2_adapter && arch-intel_gpu_bmg_g21
// REQUIRES: linux
// REQUIRES-INTEL-DRIVER: lin: 38146

// TODO: Update windows driver once available with other tests. The current
// driver restriction aligns with get_graph_multi_queue.cpp

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// RUN: %if level_zero %{%{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}

// Tests fork/join of a restricted host_task (submit_with_event + handler path)
// across two in-order queues in a native-recording SYCL Graph. Exercises the
// host task consuming a dependency event and producing a signal event.

#define GRAPH_E2E_NATIVE_RECORDING

#include "../Inputs/enqueue_func_host_task_event.cpp"
