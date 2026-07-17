// REQUIRES: aspect-usm_shared_allocations

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// RUN: %if level_zero %{%{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}

// Tests fork/join of a restricted host_task (submit_with_event + handler path)
// across two in-order queues via the command-buffer record-replay path.

#include "Inputs/enqueue_func_host_task_event.cpp"
