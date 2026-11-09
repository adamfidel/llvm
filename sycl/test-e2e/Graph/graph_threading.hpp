#pragma once

#include "graph_common.hpp"

#include <sycl/properties/all_properties.hpp>

#include <algorithm>
#include <exception>
#include <mutex>
#include <thread>
#include <vector>

/// Thread count for graph threading tests. Clamped so small machines still
/// exercise concurrency and large machines don't oversubscribe the device.
inline unsigned numTestThreads(unsigned MinThreads = 2,
                               unsigned MaxThreads = 4) {
  return std::clamp(std::thread::hardware_concurrency(), MinThreads,
                    MaxThreads);
}

/// Runs Func(ThreadIndex) on NumThreads threads which are released together by
/// a barrier to maximize contention. Blocks until all threads have joined, then
/// rethrows the first exception thrown by any thread.
///
/// Barriers that Func waits on must be passed as TestBarriers. They are aborted
/// when any thread throws, so the other threads fail instead of hanging.
template <typename FuncT, typename... BarrierTs>
void runConcurrently(unsigned NumThreads, FuncT &&Func,
                     BarrierTs &...TestBarriers) {
  Barrier SyncPoint{NumThreads};
  std::mutex ErrorMutex;
  std::exception_ptr FirstError;
  std::vector<std::thread> Threads;
  Threads.reserve(NumThreads);

  for (unsigned ThreadIndex = 0; ThreadIndex < NumThreads; ++ThreadIndex) {
    Threads.emplace_back([&, ThreadIndex] {
      SyncPoint.wait();
      try {
        Func(ThreadIndex);
      } catch (...) {
        {
          std::lock_guard<std::mutex> Lock(ErrorMutex);
          if (!FirstError)
            FirstError = std::current_exception();
        }
        (TestBarriers.abort(), ...);
      }
    });
  }

  for (std::thread &Thread : Threads)
    Thread.join();

  if (FirstError)
    std::rethrow_exception(FirstError);
}

/// Creates NumQueues queues with QueueProps sharing the context and device of
/// Q.
inline std::vector<queue> makeQueues(const queue &Q, unsigned NumQueues,
                                     const property_list &QueueProps) {
  std::vector<queue> Queues;
  Queues.reserve(NumQueues);
  for (unsigned I = 0; I < NumQueues; ++I)
    Queues.emplace_back(Q.get_context(), Q.get_device(), QueueProps);
  return Queues;
}

/// Events a submission to Q needs to run after Prev. In-order queues order
/// submissions implicitly, so none are returned and tests exercise that
/// implicit ordering rather than masking it with explicit dependencies.
inline std::vector<event> orderAfter(const queue &Q, const event &Prev) {
  if (Q.is_in_order())
    return {};
  return {Prev};
}

/// Checks that every queue in Queues is in the Expected recording state.
inline bool checkQueueStates(const std::vector<queue> &Queues,
                             exp_ext::queue_state Expected) {
  bool Passed = true;
  for (size_t I = 0; I < Queues.size(); ++I) {
    if (Queues[I].ext_oneapi_get_state() != Expected) {
      std::cerr << "Queue " << I << " has unexpected recording state"
                << std::endl;
      Passed = false;
    }
  }
  return Passed;
}

/// Runs Test(QueueProps) with an in-order queue and, unless native recording
/// is enabled (it only supports in-order queues), an out-of-order queue.
/// Returns true if every run passed.
template <typename TestT> bool runWithQueueOrderings(TestT &&Test) {
  bool Passed = Test(property_list{property::queue::in_order{}});
  if (!Passed)
    std::cerr << "Failed with in-order queue" << std::endl;

#ifndef GRAPH_E2E_NATIVE_RECORDING
  if (!Test(property_list{})) {
    std::cerr << "Failed with out-of-order queue" << std::endl;
    Passed = false;
  }
#endif

  return Passed;
}
