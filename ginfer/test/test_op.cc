#include <gtest/gtest.h>
#include <vector>
#include "ginfer/op/op.h"
#include "ginfer/tensor/tensor.h"

TEST(OpTest, AddOpCUDA) {
  using ginfer::common::DeviceType;
  using ginfer::error::StatusCode;
  using ginfer::tensor::DataType;

  ginfer::op::AddOp add_op(DeviceType::kDeviceCUDA);
  EXPECT_EQ(add_op.opType(), ginfer::op::OpType::kOpAdd);
  EXPECT_EQ(add_op.getDeviceType(), DeviceType::kDeviceCUDA);

  ginfer::tensor::Tensor a(DataType::kDataTypeFloat32, ginfer::tensor::Shape({127, 128}), DeviceType::kDeviceCPU);
  ginfer::tensor::Tensor b(DataType::kDataTypeFloat32, ginfer::tensor::Shape({127, 128}), DeviceType::kDeviceCPU);
  ginfer::tensor::Tensor c(DataType::kDataTypeFloat32, ginfer::tensor::Shape({127, 128}), DeviceType::kDeviceCPU);

  auto a_data = a.data<float>();
  auto b_data = b.data<float>();
  auto c_ref = std::vector<float>(127 * 128);
  for (int i = 0; i < 127 * 128; ++i) {
    a_data[i] = static_cast<float>(i % 100) * 0.1f;
    b_data[i] = static_cast<float>(i % 50) * 0.2f;
    c_ref[i] = a_data[i] + b_data[i];
  }

  a.toDevice(DeviceType::kDeviceCUDA);
  b.toDevice(DeviceType::kDeviceCUDA);
  c.toDevice(DeviceType::kDeviceCUDA);

  std::vector<const ginfer::tensor::Tensor*> inputs = {&a, &b};
  std::vector<ginfer::tensor::Tensor*> outputs = {&c};
  auto status = add_op.run(inputs, outputs);
  ASSERT_TRUE(status.code() == StatusCode::kSuccess) << status.msg();

  c.toDevice(DeviceType::kDeviceCPU);

  auto c_data = c.data<float>();

  for (int i = 0; i < 127 * 128; ++i) {
    EXPECT_FLOAT_EQ(c_data[i], c_ref[i]) << " at index " << i;
  }
}
