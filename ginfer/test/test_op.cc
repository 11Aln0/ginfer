#include <gtest/gtest.h>
#include <vector>
#include "ginfer/op/layer.h"
#include "ginfer/tensor/tensor.h"

TEST(LayerTest, AddLayerCUDA) {
  using ginfer::common::DeviceType;
  using ginfer::error::StatusCode;
  using ginfer::tensor::DataType;

  ginfer::op::AddLayer add_layer(DeviceType::kDeviceCUDA, "add_layer_cuda");
  EXPECT_EQ(add_layer.layerType(), ginfer::op::LayerType::kLayerAdd);
  EXPECT_EQ(add_layer.getDeviceType(), DeviceType::kDeviceCUDA);

  ginfer::tensor::Tensor a(DataType::kDataTypeFloat32, ginfer::tensor::Shape({127, 127}),
                           DeviceType::kDeviceCPU);
  ginfer::tensor::Tensor b(DataType::kDataTypeFloat32, ginfer::tensor::Shape({127, 127}),
                           DeviceType::kDeviceCPU);
  ginfer::tensor::Tensor c(DataType::kDataTypeFloat32, ginfer::tensor::Shape({127, 127}),
                           DeviceType::kDeviceCPU);

  auto a_data = a.data<float>();
  auto b_data = b.data<float>();
  auto c_ref = std::vector<float>(127 * 127);

  for (int i = 0; i < 127 * 127; ++i) {
    a_data[i] = static_cast<float>(i % 100) * 0.1f;
    b_data[i] = static_cast<float>(i % 50) * 0.2f;
    c_ref[i] = a_data[i] + b_data[i];
  }

  a.toDevice(DeviceType::kDeviceCUDA);
  b.toDevice(DeviceType::kDeviceCUDA);
  c.toDevice(DeviceType::kDeviceCUDA);

  std::vector<const ginfer::tensor::Tensor*> inputs = {&a, &b};
  auto status = add_layer.forward(inputs, &c);
  ASSERT_TRUE(status.code() == StatusCode::kSuccess) << status.msg();

  c.toDevice(DeviceType::kDeviceCPU);

  auto c_data = c.data<float>();

  for (int i = 0; i < 127 * 127; ++i) {
    EXPECT_FLOAT_EQ(c_data[i], c_ref[i]) << " at index " << i;
  }
}