#include <gtest/gtest.h>
#include <vector>
#include "ginfer/core/context.h"
#include "ginfer/core/op/op.h"
#include "ginfer/core/tensor/tensor.h"

TEST(OpTest, AddOpCUDA) {
  using ginfer::common::DeviceType;

  using ginfer::core::tensor::DataType;

  ginfer::core::op::AddOp add_op(DeviceType::kDeviceCUDA);
  EXPECT_EQ(add_op.opType(), ginfer::core::op::OpType::kOpAdd);
  EXPECT_EQ(add_op.getDeviceType(), DeviceType::kDeviceCUDA);

  auto a_res = ginfer::core::tensor::Tensor::create(
      DataType::kDataTypeFloat32, ginfer::core::tensor::Shape({127, 128}), DeviceType::kDeviceCPU);
  ASSERT_TRUE(a_res.ok()) << a_res.err();
  auto a = a_res.value();

  auto b_res = ginfer::core::tensor::Tensor::create(
      DataType::kDataTypeFloat32, ginfer::core::tensor::Shape({127, 128}), DeviceType::kDeviceCPU);
  ASSERT_TRUE(b_res.ok()) << b_res.err();
  auto b = b_res.value();

  auto c_res = ginfer::core::tensor::Tensor::create(
      DataType::kDataTypeFloat32, ginfer::core::tensor::Shape({127, 128}), DeviceType::kDeviceCPU);
  ASSERT_TRUE(c_res.ok()) << c_res.err();
  auto c = c_res.value();

  auto a_data = a->data<float>();
  auto b_data = b->data<float>();
  auto c_ref = std::vector<float>(127 * 128);
  for (int i = 0; i < 127 * 128; ++i) {
    a_data[i] = static_cast<float>(i % 100) * 0.1f;
    b_data[i] = static_cast<float>(i % 50) * 0.2f;
    c_ref[i] = a_data[i] + b_data[i];
  }

  auto a_dev_res = a->toDevice(DeviceType::kDeviceCUDA);
  ASSERT_TRUE(a_dev_res.ok()) << a_dev_res.err();
  a = a_dev_res.value();
  auto b_dev_res = b->toDevice(DeviceType::kDeviceCUDA);
  ASSERT_TRUE(b_dev_res.ok()) << b_dev_res.err();
  b = b_dev_res.value();
  auto c_dev_res = c->toDevice(DeviceType::kDeviceCUDA);
  ASSERT_TRUE(c_dev_res.ok()) << c_dev_res.err();
  c = c_dev_res.value();

  auto dev_ctx = ginfer::common::DeviceContext::create(DeviceType::kDeviceCUDA);
  std::vector<const ginfer::core::tensor::Tensor*> inputs = {a.get(), b.get()};
  std::vector<ginfer::core::tensor::Tensor*> outputs = {c.get()};
  auto status = add_op.run(ginfer::core::InferContext{}.setDeviceContext(dev_ctx), inputs, outputs);
  ASSERT_TRUE(status.ok()) << status.err();

  auto c_cpu_res = c->toDevice(DeviceType::kDeviceCPU);
  ASSERT_TRUE(c_cpu_res.ok()) << c_cpu_res.err();
  c = c_cpu_res.value();

  auto c_data = c->data<float>();

  for (int i = 0; i < 127 * 128; ++i) {
    EXPECT_FLOAT_EQ(c_data[i], c_ref[i]) << " at index " << i;
  }
}
