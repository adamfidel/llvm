//==------- reusable_events.cpp --- SYCL reusable events -------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "detail/context_impl.hpp"
#include "detail/event_impl.hpp"
#include "detail/graph/graph_impl.hpp"
#include "detail/queue_impl.hpp"
#include <sycl/detail/ur.hpp>
#include <sycl/ext/oneapi/experimental/reusable_events.hpp>

namespace sycl {
inline namespace _V1 {
namespace ext::oneapi::experimental {

namespace detail {

__SYCL_EXPORT sycl::event make_event(const sycl::context &ctxt,
                                     uint32_t Flags) {
  const bool EnableProfiling = Flags & make_event_flag_enable_profiling;
  const bool EnableIPC = Flags & make_event_flag_enable_ipc;

  // enable_profiling and enable_ipc are mutually exclusive.
  if (EnableProfiling && EnableIPC) {
    throw sycl::exception(
        sycl::make_error_code(errc::invalid),
        "The enable_profiling and enable_ipc properties cannot both be set "
        "when creating an event.");
  }

  detail::context_impl &ContextImpl = *sycl::detail::getSyclObjImpl(ctxt);

  if (EnableProfiling && !ContextImpl.supportsEventProfiling()) {
    throw sycl::exception(sycl::make_error_code(errc::feature_not_supported),
                          "Context does not support per-event profiling.");
  }

  // enable_ipc requires every device in the context to support IPC events.
  if (EnableIPC && !ContextImpl.supportsIPCEvents()) {
    throw sycl::exception(sycl::make_error_code(errc::feature_not_supported),
                          "Not all devices in the context support "
                          "aspect::ext_oneapi_ipc_event.");
  }

  sycl::event RetEvent{};
  detail::event_impl &EventImpl = *sycl::detail::getSyclObjImpl(RetEvent);
  EventImpl.setContextImpl(ContextImpl);
  EventImpl.setProfilingEnabled(EnableProfiling);
  EventImpl.setIPCEnabled(EnableIPC);

  // The backend UR event is created lazily on first signal or first
  // ipc::event::get.
  return RetEvent;
}

static void CheckEventAndThrow(detail::event_impl &EventImpl,
                               detail::context_impl &ContextImpl) {
  if (EventImpl.isHost()) {
    throw sycl::exception(sycl::make_error_code(errc::invalid),
                          "Host events cannot be enqueued for waiting.");
  }

  // Current limitation:
  // The queue and an event need to be in the same context. The reason
  // is, that cross-context dependencies use host tasks, and the wait
  // command might be queued in the runtime. This flow is currently
  // not supported by the Reusable Events APIs.
  if (&EventImpl.getContextImpl() != &ContextImpl) {
    throw sycl::exception(sycl::make_error_code(errc::invalid),
                          "Event context must match the queue context.");
  }
}

/// An event which was enqueued for signaling while a queue was recording a
/// graph is not signaled by the backend at all: the signal is a node of that
/// graph, so a wait on the event has to become an edge from that node. Prepares
/// such a wait, or rejects it if it cannot be expressed as an edge.
static void PrepareGraphWaitAndThrow(detail::queue_impl &QueueImpl,
                                     detail::event_impl &EventImpl) {
  auto EventGraph = EventImpl.getCommandGraph();
  auto QueueGraph = QueueImpl.getCommandGraph();

  if (QueueGraph && QueueGraph != EventGraph) {
    throw sycl::exception(
        sycl::make_error_code(errc::invalid),
        "An event can only be enqueued for waiting on a queue which is "
        "recording a graph if it was enqueued for signaling on a queue "
        "recording the same graph.");
  }

  // Transitive queue recording: the wait is what pulls this queue into the
  // graph, exactly as it does for a dependency taken through depends_on.
  if (EventGraph && !QueueGraph)
    EventGraph->beginRecording(QueueImpl);
}

} // namespace detail

__SYCL_EXPORT void enqueue_wait_event(sycl::queue q, const event &evt) {
  detail::queue_impl &QueueImpl = *sycl::detail::getSyclObjImpl(q);
  detail::event_impl &EventImpl = *sycl::detail::getSyclObjImpl(evt);

  detail::CheckEventAndThrow(EventImpl, QueueImpl.getContextImpl());
  detail::PrepareGraphWaitAndThrow(QueueImpl, EventImpl);

  QueueImpl.submit_barrier_direct_without_event(
      sycl::span<const event>(&evt, 1), detail::CGType::BarrierWaitlist,
      detail::code_location::current());
}

__SYCL_EXPORT void enqueue_wait_events(sycl::queue q,
                                       const std::vector<event> &evts) {
  detail::queue_impl &QueueImpl = *sycl::detail::getSyclObjImpl(q);

  for (const sycl::event &evt : evts) {
    detail::event_impl &EventImpl = *sycl::detail::getSyclObjImpl(evt);
    detail::CheckEventAndThrow(EventImpl, QueueImpl.getContextImpl());
    detail::PrepareGraphWaitAndThrow(QueueImpl, EventImpl);
  }

  QueueImpl.submit_barrier_direct_without_event(
      evts, detail::CGType::BarrierWaitlist, detail::code_location::current());
}

__SYCL_EXPORT void enqueue_signal_event(sycl::queue q, event &evt) {
  detail::queue_impl &QueueImpl = *sycl::detail::getSyclObjImpl(q);
  detail::event_impl &EventImpl = *sycl::detail::getSyclObjImpl(evt);

  if (EventImpl.isInterop()) {
    throw sycl::exception(
        sycl::make_error_code(errc::runtime),
        "Enqueueing an interop event for signaling is not supported.");
  }

  // While recording, the signal becomes an empty graph node rather than a
  // backend signal, so nothing ever signals the handle exported to a peer
  // process.
  if (QueueImpl.hasCommandGraph() && EventImpl.isIPCEnabled()) {
    throw sycl::exception(sycl::make_error_code(errc::invalid),
                          "An IPC-enabled event cannot be enqueued for "
                          "signaling on a queue which is recording a graph.");
  }

  detail::CheckEventAndThrow(EventImpl, QueueImpl.getContextImpl());

  // An IPC event cannot be signaled on a profiling-enabled queue.
  if (EventImpl.isIPCEnabled() && QueueImpl.MIsProfilingEnabled) {
    throw sycl::exception(
        sycl::make_error_code(errc::invalid),
        "An IPC-enabled event cannot be signaled on a queue that has "
        "profiling enabled.");
  }

  QueueImpl.submit_barrier_direct_without_event(
      {}, detail::CGType::Barrier, detail::code_location::current(),
      sycl::detail::getSyclObjImpl(evt));
}

} // namespace ext::oneapi::experimental
} // namespace _V1
} // namespace sycl
