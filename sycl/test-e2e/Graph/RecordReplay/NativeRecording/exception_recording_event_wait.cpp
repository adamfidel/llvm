// REQUIRES: level_zero_v2_adapter && arch-intel_gpu_bmg_g21
// REQUIRES: linux
// REQUIRES-INTEL-DRIVER: lin: 38583

// TODO: Add minimum Windows driver version once available

// The native recording path can only report the spec-mandated errc::invalid
// once the underlying runtime exposes the granular graph capture error codes.
// Remove this XFAIL when that support lands.
// XFAIL: level_zero_v2_adapter

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if level_zero %{%{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}

#define GRAPH_E2E_NATIVE_RECORDING

#include "../../Inputs/exception_recording_event_wait.cpp"
