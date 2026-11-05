#pragma once

#include "graph_common.hpp"

#include <sycl/properties/all_properties.hpp>

#include <algorithm>
#include <exception>
#include <thread>
#include <vector>

/// Thread count for graph threading tests. Clamped so small machines still
/// exercise concurrency and large machines don't oversubscribe the device.
inline unsigned numTestThreads(unsigned MinThreads = 2,
                               unsigned MaxThreads = 8) {
  return std::clamp(std::thread::hardware_concurrency(), MinThreads,
                    MaxThreads);
}

/// Runs Func(ThreadIndex) on NumThreads threads which are released together by
/// a barrier to maximize contention. Blocks until all threads have joined, then
/// rethrows the first exception thrown by any thread.
template <typename FuncT>
void runConcurrently(unsigned NumThreads, FuncT &&Func) {
  Barrier SyncPoint{NumThreads};
  std::vector<std::exception_ptr> Errors(NumThreads);
  std::vector<std::thread> Threads;
  Threads.reserve(NumThreads);

  for (unsigned ThreadIndex = 0; ThreadIndex < NumThreads; ++ThreadIndex) {
    Threads.emplace_back([&, ThreadIndex] {
      SyncPoint.wait();
      try {
        Func(ThreadIndex);
      } catch (...) {
        Errors[ThreadIndex] = std::current_exception();
      }
    });
  }

  for (std::thread &Thread : Threads)
    Thread.join();

  for (std::exception_ptr &Error : Errors)
    if (Error)
      std::rethrow_exception(Error);
}

/// Creates NumQueues in-order queues sharing the context and device of Q.
inline std::vector<queue> makeInOrderQueues(const queue &Q,
                                            unsigned NumQueues) {
  std::vector<queue> Queues;
  Queues.reserve(NumQueues);
  for (unsigned I = 0; I < NumQueues; ++I)
    Queues.emplace_back(Q.get_context(), Q.get_device(),
                        property::queue::in_order{});
  return Queues;
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
