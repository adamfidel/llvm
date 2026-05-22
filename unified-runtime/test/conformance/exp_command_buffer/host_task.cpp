// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "fixtures.h"

struct urCommandBufferHostTaskTest
    : uur::command_buffer::urCommandBufferExpTest {
  void SetUp() override {
    UUR_RETURN_ON_FATAL_FAILURE(
        uur::command_buffer::urCommandBufferExpTest::SetUp());

    ASSERT_SUCCESS(urQueueCreate(context, device, nullptr, &queue));
  }

  void TearDown() override {
    if (queue) {
      EXPECT_SUCCESS(urQueueRelease(queue));
    }
    UUR_RETURN_ON_FATAL_FAILURE(
        uur::command_buffer::urCommandBufferExpTest::TearDown());
  }

  ur_queue_handle_t queue = nullptr;
};

UUR_INSTANTIATE_DEVICE_TEST_SUITE(urCommandBufferHostTaskTest);

namespace {
void SetFlagTask(void *data) { *static_cast<int *>(data) = 1; }
} // namespace

// Verify that the host task callback is not invoked until the command buffer
// is actually executed (not at append time, not at finalize time).
TEST_P(urCommandBufferHostTaskTest, DeferredExecution) {
  int flag = 0;

  ASSERT_SUCCESS(urCommandBufferAppendHostTaskExp(
      cmd_buf_handle, SetFlagTask, &flag, nullptr, 0, nullptr, 0, nullptr,
      nullptr, nullptr, nullptr));
  ASSERT_EQ(flag, 0) << "Host task must not run at append time";

  ASSERT_SUCCESS(urCommandBufferFinalizeExp(cmd_buf_handle));
  ASSERT_EQ(flag, 0) << "Host task must not run at finalize time";

  ASSERT_SUCCESS(
      urEnqueueCommandBufferExp(queue, cmd_buf_handle, 0, nullptr, nullptr));
  ASSERT_SUCCESS(urQueueFinish(queue));
  ASSERT_EQ(flag, 1) << "Host task must have run after execution";
}

// Verify sync point dependencies: host task waits on a preceding USM fill,
// then a subsequent memcpy depends on the host task's sync point.
struct urCommandBufferHostTaskSyncPointTest : uur::urQueueTest {
  void SetUp() override {
    UUR_RETURN_ON_FATAL_FAILURE(uur::urQueueTest::SetUp());

    UUR_RETURN_ON_FATAL_FAILURE(
        uur::command_buffer::checkCommandBufferSupport(device));

    ur_exp_command_buffer_desc_t desc{UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_DESC,
                                      nullptr, false, false, false};
    ASSERT_SUCCESS(urCommandBufferCreateExp(context, device, &desc, &cmd_buf));

    ASSERT_SUCCESS(
        urUSMHostAlloc(context, nullptr, nullptr, allocation_size, &host_ptr));
    ASSERT_NE(host_ptr, nullptr);
  }

  void TearDown() override {
    if (cmd_buf) {
      EXPECT_SUCCESS(urCommandBufferReleaseExp(cmd_buf));
    }
    if (host_ptr) {
      EXPECT_SUCCESS(urUSMFree(context, host_ptr));
    }
    UUR_RETURN_ON_FATAL_FAILURE(uur::urQueueTest::TearDown());
  }

  ur_exp_command_buffer_handle_t cmd_buf = nullptr;
  void *host_ptr = nullptr;
  static constexpr size_t num_elements = 64;
  static constexpr size_t allocation_size = num_elements * sizeof(uint32_t);
};

UUR_INSTANTIATE_DEVICE_TEST_SUITE(urCommandBufferHostTaskSyncPointTest);

namespace {
void DoubleElements(void *data) {
  uint32_t *arr = static_cast<uint32_t *>(data);
  for (size_t i = 0; i < urCommandBufferHostTaskSyncPointTest::num_elements;
       i++) {
    arr[i] *= 2;
  }
}
} // namespace

// Fill buffer -> host task doubles values -> verify result
TEST_P(urCommandBufferHostTaskSyncPointTest, Dependencies) {
  uint32_t fill_val = 21;

  ur_exp_command_buffer_sync_point_t fill_sync;
  ASSERT_SUCCESS(urCommandBufferAppendUSMFillExp(
      cmd_buf, host_ptr, &fill_val, sizeof(fill_val), allocation_size, 0,
      nullptr, 0, nullptr, &fill_sync, nullptr, nullptr));

  ur_exp_command_buffer_sync_point_t host_task_sync;
  ASSERT_SUCCESS(urCommandBufferAppendHostTaskExp(
      cmd_buf, DoubleElements, host_ptr, nullptr, 1, &fill_sync, 0, nullptr,
      &host_task_sync, nullptr, nullptr));

  ASSERT_SUCCESS(urCommandBufferFinalizeExp(cmd_buf));
  ASSERT_SUCCESS(
      urEnqueueCommandBufferExp(queue, cmd_buf, 0, nullptr, nullptr));
  ASSERT_SUCCESS(urQueueFinish(queue));

  uint32_t *result = static_cast<uint32_t *>(host_ptr);
  for (size_t i = 0; i < num_elements; i++) {
    ASSERT_EQ(result[i], 42u) << "Mismatch at index " << i;
  }
}

// In-order command buffer: host task runs in sequence without explicit
// sync points.
struct urCommandBufferHostTaskInOrderTest : uur::urQueueTest {
  void SetUp() override {
    UUR_RETURN_ON_FATAL_FAILURE(uur::urQueueTest::SetUp());

    UUR_RETURN_ON_FATAL_FAILURE(
        uur::command_buffer::checkCommandBufferSupport(device));

    ur_exp_command_buffer_desc_t desc{UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_DESC,
                                      nullptr, false, true, false};
    ASSERT_SUCCESS(urCommandBufferCreateExp(context, device, &desc, &cmd_buf));

    ASSERT_SUCCESS(
        urUSMHostAlloc(context, nullptr, nullptr, allocation_size, &host_ptr));
    ASSERT_NE(host_ptr, nullptr);
  }

  void TearDown() override {
    if (cmd_buf) {
      EXPECT_SUCCESS(urCommandBufferReleaseExp(cmd_buf));
    }
    if (host_ptr) {
      EXPECT_SUCCESS(urUSMFree(context, host_ptr));
    }
    UUR_RETURN_ON_FATAL_FAILURE(uur::urQueueTest::TearDown());
  }

  ur_exp_command_buffer_handle_t cmd_buf = nullptr;
  void *host_ptr = nullptr;
  static constexpr size_t num_elements = 64;
  static constexpr size_t allocation_size = num_elements * sizeof(uint32_t);
};

UUR_INSTANTIATE_DEVICE_TEST_SUITE(urCommandBufferHostTaskInOrderTest);

namespace {
void IncrementElements(void *data) {
  uint32_t *arr = static_cast<uint32_t *>(data);
  for (size_t i = 0; i < urCommandBufferHostTaskInOrderTest::num_elements;
       i++) {
    arr[i] += 1;
  }
}
} // namespace

// Fill -> host task increment -> host task increment -> verify (fill+2)
TEST_P(urCommandBufferHostTaskInOrderTest, Success) {
  uint32_t fill_val = 40;

  ASSERT_SUCCESS(urCommandBufferAppendUSMFillExp(
      cmd_buf, host_ptr, &fill_val, sizeof(fill_val), allocation_size, 0,
      nullptr, 0, nullptr, nullptr, nullptr, nullptr));

  ASSERT_SUCCESS(urCommandBufferAppendHostTaskExp(
      cmd_buf, IncrementElements, host_ptr, nullptr, 0, nullptr, 0, nullptr,
      nullptr, nullptr, nullptr));

  ASSERT_SUCCESS(urCommandBufferAppendHostTaskExp(
      cmd_buf, IncrementElements, host_ptr, nullptr, 0, nullptr, 0, nullptr,
      nullptr, nullptr, nullptr));

  ASSERT_SUCCESS(urCommandBufferFinalizeExp(cmd_buf));
  ASSERT_SUCCESS(
      urEnqueueCommandBufferExp(queue, cmd_buf, 0, nullptr, nullptr));
  ASSERT_SUCCESS(urQueueFinish(queue));

  uint32_t *result = static_cast<uint32_t *>(host_ptr);
  for (size_t i = 0; i < num_elements; i++) {
    ASSERT_EQ(result[i], 42u) << "Mismatch at index " << i;
  }
}
