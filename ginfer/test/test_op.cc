#include <gtest/gtest.h>
#include <vector>
#include "ginfer/op/op.h"
#include "ginfer/tensor/tensor.h"

TEST(OpTest, AddOpCUDA) {
  using ginfer::common::DeviceType;

  using ginfer::tensor::DataType;

  ginfer::op::AddOp add_op(DeviceType::kDeviceCUDA);
  EXPECT_EQ(add_op.opType(), ginfer::op::OpType::kOpAdd);
  EXPECT_EQ(add_op.getDeviceType(), DeviceType::kDeviceCUDA);

  auto a_res = ginfer::tensor::Tensor::create(DataType::kDataTypeFloat32, ginfer::tensor::Shape({127, 128}), DeviceType::kDeviceCPU);
  ASSERT_TRUE(a_res.ok()) << a_res.err();
  auto a = a_res.value();

  auto b_res = ginfer::tensor::Tensor::create(DataType::kDataTypeFloat32, ginfer::tensor::Shape({127, 128}), DeviceType::kDeviceCPU);
  ASSERT_TRUE(b_res.ok()) << b_res.err();
  auto b = b_res.value();

  auto c_res = ginfer::tensor::Tensor::create(DataType::kDataTypeFloat32, ginfer::tensor::Shape({127, 128}), DeviceType::kDeviceCPU);
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

  a->toDevice(DeviceType::kDeviceCUDA);
  b->toDevice(DeviceType::kDeviceCUDA);
  c->toDevice(DeviceType::kDeviceCUDA);

  std::vector<const ginfer::tensor::Tensor*> inputs = {a.get(), b.get()};
  std::vector<ginfer::tensor::Tensor*> outputs = {c.get()};
  auto status = add_op.run(inputs, outputs);
  ASSERT_TRUE(status.ok()) << status.err();

  c->toDevice(DeviceType::kDeviceCPU);

  auto c_data = c->data<float>();

  for (int i = 0; i < 127 * 128; ++i) {
    EXPECT_FLOAT_EQ(c_data[i], c_ref[i]) << " at index " << i;
  }
}
