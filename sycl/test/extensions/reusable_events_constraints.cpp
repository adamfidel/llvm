// RUN: %clangxx -fsyntax-only -fsycl -Xclang -verify -Xclang -verify-ignore-unexpected=note %s

// make_event only accepts event properties and the enqueue functions only
// accept enqueue properties.

#include <sycl/ext/oneapi/experimental/annotated_ptr/annotated_ptr_properties.hpp>
#include <sycl/ext/oneapi/experimental/reusable_events.hpp>

namespace syclex = sycl::ext::oneapi::experimental;

void test(sycl::queue q, sycl::event e, std::vector<sycl::event> v) {
  sycl::context ctx = q.get_context();

  // OK
  syclex::make_event();
  syclex::make_event(ctx);
  syclex::make_event(ctx, syclex::properties{});
  syclex::make_event(ctx, syclex::enable_profiling{});
  syclex::make_event(ctx, syclex::properties{syclex::enable_ipc{}});
  syclex::make_event(
      ctx, syclex::properties{
               syclex::enable_profiling{},
               syclex::event_mode{syclex::event_mode_enum::low_power}});
  syclex::make_event(syclex::properties{syclex::enable_profiling{}});
  syclex::make_event(syclex::enable_ipc{});

  // expected-error@+1 {{no matching function for call to 'make_event'}}
  syclex::make_event(
      ctx, syclex::properties{syclex::usm_kind<sycl::usm::alloc::host>});
  // expected-error@+1 {{no matching function for call to 'make_event'}}
  syclex::make_event(ctx, syclex::usm_kind<sycl::usm::alloc::host>);
  // expected-error@+1 {{no matching function for call to 'make_event'}}
  syclex::make_event(
      syclex::properties{syclex::usm_kind<sycl::usm::alloc::host>});
  // expected-error@+1 {{no matching function for call to 'make_event'}}
  syclex::make_event(syclex::usm_kind<sycl::usm::alloc::host>);
  // expected-error@+1 {{no matching function for call to 'make_event'}}
  syclex::make_event(ctx, syclex::properties{syclex::graph_external{}});
  // expected-error@+1 {{no matching function for call to 'make_event'}}
  syclex::make_event(syclex::graph_external{});

  // OK
  syclex::enqueue_wait_event(q, e);
  syclex::enqueue_wait_event(q, e, syclex::properties{});
  syclex::enqueue_wait_events(q, v);
  syclex::enqueue_wait_events(q, v, syclex::properties{});
  syclex::enqueue_signal_event(q, e);
  syclex::enqueue_signal_event(q, e, syclex::properties{});
  syclex::enqueue_wait_event(q, e,
                             syclex::properties{syclex::graph_external{}});
  syclex::enqueue_wait_events(q, v,
                              syclex::properties{syclex::graph_external{}});
  syclex::enqueue_signal_event(q, e,
                               syclex::properties{syclex::graph_external{}});
  syclex::enqueue_wait_event(q, e, syclex::graph_external{});
  syclex::enqueue_wait_events(q, v, syclex::graph_external{});
  syclex::enqueue_signal_event(q, e, syclex::graph_external{});

  // expected-error@+1 {{no matching function for call to 'enqueue_wait_event'}}
  syclex::enqueue_wait_event(q, e,
                             syclex::properties{syclex::enable_profiling{}});
  // expected-error@+1 {{no matching function for call to 'enqueue_wait_events'}}
  syclex::enqueue_wait_events(q, v,
                              syclex::properties{syclex::enable_profiling{}});
  // expected-error@+1 {{no matching function for call to 'enqueue_signal_event'}}
  syclex::enqueue_signal_event(q, e,
                               syclex::properties{syclex::enable_profiling{}});
  // expected-error@+1 {{no matching function for call to 'enqueue_wait_event'}}
  syclex::enqueue_wait_event(q, e, syclex::enable_profiling{});
  // expected-error@+1 {{no matching function for call to 'enqueue_wait_events'}}
  syclex::enqueue_wait_events(q, v, syclex::enable_profiling{});
  // expected-error@+1 {{no matching function for call to 'enqueue_signal_event'}}
  syclex::enqueue_signal_event(q, e, syclex::enable_profiling{});
  // expected-error@+1 {{no matching function for call to 'enqueue_wait_event'}}
  syclex::enqueue_wait_event(
      q, e,
      syclex::properties{syclex::graph_external{}, syclex::enable_profiling{}});
}
