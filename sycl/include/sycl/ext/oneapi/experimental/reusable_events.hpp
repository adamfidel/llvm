//==------- reusable_events.hpp --- SYCL reusable events -------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/context.hpp>
#include <sycl/device.hpp>
#include <sycl/event.hpp>
#include <sycl/ext/oneapi/experimental/detail/ipc_common.hpp>
#include <sycl/ext/oneapi/properties/properties.hpp>
#include <sycl/platform.hpp>
#include <sycl/queue.hpp>

#include <cstdint>

namespace sycl {
inline namespace _V1 {
namespace ext::oneapi::experimental {

struct enable_profiling
    : detail::run_time_property_key<enable_profiling,
                                    detail::PropKind::EnableProfiling> {
  constexpr enable_profiling(bool enable = true) : value(enable) {}
  bool value;
};

using enable_profiling_key = enable_profiling;

inline bool operator==(const enable_profiling &lhs,
                       const enable_profiling &rhs) {
  return lhs.value == rhs.value;
}
inline bool operator!=(const enable_profiling &lhs,
                       const enable_profiling &rhs) {
  return !(lhs == rhs);
}

/// Marks an enqueued signal or wait operation as external to a graph being
/// recorded on the queue. An external wait becomes a graph input which the host
/// must signal before each replay, and an external signal becomes a graph
/// output which can be observed from outside the graph.
struct graph_external
    : detail::run_time_property_key<graph_external,
                                    detail::PropKind::GraphExternal> {
  constexpr graph_external(bool enable = true) : value(enable) {}
  bool value;
};

using graph_external_key = graph_external;

inline bool operator==(const graph_external &lhs, const graph_external &rhs) {
  return lhs.value == rhs.value;
}
inline bool operator!=(const graph_external &lhs, const graph_external &rhs) {
  return !(lhs == rhs);
}

template <>
struct is_property_key_of<enable_profiling_key, sycl::event> : std::true_type {
};

template <>
struct is_property_key_of<enable_ipc_key, sycl::event> : std::true_type {};

template <>
struct is_property_key_of<graph_external_key, sycl::event> : std::true_type {};

namespace detail {
enum make_event_flags : uint32_t {
  make_event_flag_enable_profiling = 1u << 0,
  make_event_flag_enable_ipc = 1u << 1,
};

__SYCL_EXPORT sycl::event make_event(const sycl::context &ctxt, uint32_t Flags);

template <typename PropertyListT>
uint32_t getMakeEventFlags(const PropertyListT &props) {
  uint32_t Flags = 0;
  if constexpr (PropertyListT::template has_property<enable_profiling_key>()) {
    if (props.template get_property<enable_profiling_key>().value)
      Flags |= make_event_flag_enable_profiling;
  }
  if constexpr (PropertyListT::template has_property<enable_ipc_key>()) {
    if (props.template get_property<enable_ipc_key>().value)
      Flags |= make_event_flag_enable_ipc;
  }
  return Flags;
}

enum enqueue_event_flags : uint32_t {
  enqueue_event_flag_graph_external = 1u << 0,
};

__SYCL_EXPORT void enqueue_wait_event(sycl::queue q, const event &evt,
                                      uint32_t Flags);
__SYCL_EXPORT void enqueue_wait_events(sycl::queue q,
                                       const std::vector<event> &evts,
                                       uint32_t Flags);
__SYCL_EXPORT void enqueue_signal_event(sycl::queue q, event &evt,
                                        uint32_t Flags);

template <typename PropertyListT>
uint32_t getEnqueueEventFlags(const PropertyListT &props) {
  uint32_t Flags = 0;
  if constexpr (PropertyListT::template has_property<graph_external_key>()) {
    if (props.template get_property<graph_external_key>().value)
      Flags |= enqueue_event_flag_graph_external;
  }
  return Flags;
}
} // namespace detail

template <typename PropertyListT = empty_properties_t>
inline sycl::event make_event(const sycl::context &ctxt,
                              PropertyListT props = {}) {
  static_assert(is_property_list_v<PropertyListT>,
                "Props must be a sycl::ext::oneapi::experimental::properties");

  return detail::make_event(ctxt, detail::getMakeEventFlags(props));
}

template <typename PropertyListT = empty_properties_t>
inline sycl::event make_event(PropertyListT props = {}) {
  sycl::device Dev;
  sycl::context Ctx = Dev.get_platform().khr_get_default_context();
  return make_event(Ctx, props);
}

__SYCL_EXPORT void enqueue_wait_event(sycl::queue q, const event &evt);
__SYCL_EXPORT void enqueue_wait_events(sycl::queue q,
                                       const std::vector<event> &evts);
__SYCL_EXPORT void enqueue_signal_event(sycl::queue q, event &evt);

template <typename PropertyListT = empty_properties_t>
inline void enqueue_wait_event(sycl::queue q, const event &evt,
                               PropertyListT props = {}) {
  static_assert(is_property_list_v<PropertyListT>,
                "Props must be a sycl::ext::oneapi::experimental::properties");

  detail::enqueue_wait_event(q, evt, detail::getEnqueueEventFlags(props));
}

template <typename PropertyListT = empty_properties_t>
inline void enqueue_wait_events(sycl::queue q, const std::vector<event> &evts,
                                PropertyListT props = {}) {
  static_assert(is_property_list_v<PropertyListT>,
                "Props must be a sycl::ext::oneapi::experimental::properties");

  detail::enqueue_wait_events(q, evts, detail::getEnqueueEventFlags(props));
}

template <typename PropertyListT = empty_properties_t>
inline void enqueue_signal_event(sycl::queue q, event &evt,
                                 PropertyListT props = {}) {
  static_assert(is_property_list_v<PropertyListT>,
                "Props must be a sycl::ext::oneapi::experimental::properties");

  detail::enqueue_signal_event(q, evt, detail::getEnqueueEventFlags(props));
}

} // namespace ext::oneapi::experimental
} // namespace _V1
} // namespace sycl
