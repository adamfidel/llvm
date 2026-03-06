//==--------- level_zero.cpp - SYCL Level-Zero backend ---------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/adapter_impl.hpp>
#include <detail/graph/graph_impl.hpp>
#include <detail/platform_impl.hpp>
#include <detail/queue_impl.hpp>
#include <detail/ur.hpp>
#include <sycl/backend_types.hpp>
#include <sycl/ext/oneapi/backend/level_zero.hpp>

namespace sycl {
inline namespace _V1 {
namespace ext::oneapi::level_zero::detail {
using namespace sycl::detail;

__SYCL_EXPORT device make_device(const platform &Platform,
                                 ur_native_handle_t NativeHandle) {
  adapter_impl &Adapter = ur::getAdapter<backend::ext_oneapi_level_zero>();
  // Create UR device first.
  ur_device_handle_t UrDevice;
  Adapter.call<UrApiKind::urDeviceCreateWithNativeHandle>(
      NativeHandle, Adapter.getUrAdapter(), nullptr, &UrDevice);

  return detail::createSyclObjFromImpl<device>(
      getSyclObjImpl(Platform)->getOrMakeDeviceImpl(UrDevice));
}

} // namespace ext::oneapi::level_zero::detail

// Specialization of sycl::get_native for modifiable command graph.
template <>
__SYCL_EXPORT auto
get_native<backend::ext_oneapi_level_zero,
           ext::oneapi::experimental::command_graph<
               ext::oneapi::experimental::graph_state::modifiable>>(
    const ext::oneapi::experimental::command_graph<
        ext::oneapi::experimental::graph_state::modifiable> &Obj)
    -> backend_return_t<
        backend::ext_oneapi_level_zero,
        ext::oneapi::experimental::command_graph<
            ext::oneapi::experimental::graph_state::modifiable>> {
  // Cast to detail base class to access impl
  auto &DetailGraph = static_cast<
      const ext::oneapi::experimental::detail::modifiable_command_graph &>(Obj);
  // Use getSyclObjImpl to access impl through friend mechanism (ABI-safe)
  auto GraphImpl = sycl::detail::getSyclObjImpl(DetailGraph);

  // Get the native UR graph handle
  ur_exp_graph_handle_t URHandle = GraphImpl->getNativeGraphHandle();

  // Check if native recording is enabled
  if (URHandle == nullptr) {
    throw sycl::exception(
        make_error_code(errc::feature_not_supported),
        "get_native() is only supported for graphs created with native "
        "recording enabled. Set SYCL_GRAPH_ENABLE_NATIVE_RECORDING=1 and use "
        "immediate command lists.");
  }

  // Cast UR handle to Level Zero native graph handle
  return reinterpret_cast<ze_graph_handle_t>(URHandle);
}

// Specialization of sycl::get_native for executable command graph.
template <>
__SYCL_EXPORT auto
get_native<backend::ext_oneapi_level_zero,
           ext::oneapi::experimental::command_graph<
               ext::oneapi::experimental::graph_state::executable>>(
    const ext::oneapi::experimental::command_graph<
        ext::oneapi::experimental::graph_state::executable> &Obj)
    -> backend_return_t<
        backend::ext_oneapi_level_zero,
        ext::oneapi::experimental::command_graph<
            ext::oneapi::experimental::graph_state::executable>> {
  // Cast to detail base class to access impl
  auto &DetailGraph = static_cast<
      const ext::oneapi::experimental::detail::executable_command_graph &>(Obj);
  // Use getSyclObjImpl to access impl through friend mechanism (ABI-safe)
  auto ExecGraphImpl = sycl::detail::getSyclObjImpl(DetailGraph);

  // Get the native UR executable graph handle
  ur_exp_executable_graph_handle_t URHandle =
      ExecGraphImpl->getNativeExecutableGraphHandle();

  // Check if native recording was enabled
  if (URHandle == nullptr) {
    throw sycl::exception(
        make_error_code(errc::feature_not_supported),
        "get_native() is only supported for executable graphs created from "
        "graphs with native recording enabled. Set "
        "SYCL_GRAPH_ENABLE_NATIVE_RECORDING=1 and use immediate command "
        "lists.");
  }

  // Cast UR handle to Level Zero native executable graph handle
  return reinterpret_cast<ze_executable_graph_handle_t>(URHandle);
}

} // namespace _V1
} // namespace sycl
