#pragma once

#include <string>
#include <vector>

#include "ginfer/common/device.h"
#include "ginfer/common/errors.h"
#include "ginfer/tensor/tensor.h"

namespace ginfer::op {

using ginfer::common::DeviceType;
using ginfer::common::Status;

enum class LayerType : uint8_t {
  kLayerUnknown = 0,
  kLayerLinear = 1,
  kLayerEncode = 2,
  kLayerEmbedding = 3,
  kLayerRMSNorm = 4,
  kLayerMatmul = 5,
  kLayerROPE = 6,
  kLayerMHA = 7,
  kLayerSoftmax = 8,
  kLayerAdd = 9,
  kLayerSwiGLU = 10,
};

class BaseLayer {
 public:
  explicit BaseLayer(DeviceType dev_type, LayerType layer_type, std::string layer_name);

  LayerType layerType() const { return layer_type_; }

  DeviceType devType() const { return dev_type_; }

  virtual Status forward(const tensor::Tensor& input1, const tensor::Tensor& output1) = 0;

  virtual Status forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                         const tensor::Tensor& output1) = 0;

  virtual Status forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                         const tensor::Tensor& input3, const tensor::Tensor& output1) = 0;

  virtual Status forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                         const tensor::Tensor& input3, const tensor::Tensor& input4,
                         const tensor::Tensor& output1) = 0;

  virtual Status forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                         const tensor::Tensor& input3, const tensor::Tensor& input4,
                         const tensor::Tensor& input5, const tensor::Tensor& output1) = 0;

  virtual Status to_dev(DeviceType dev_type) = 0;

 protected:
  virtual Status forward() = 0;

  virtual void set_input(int32_t idx, const tensor::Tensor& input) = 0;

  virtual void set_output(int32_t idx, const tensor::Tensor& output) = 0;

 private:
  DeviceType dev_type_ = DeviceType::kDeviceUnknown;
  LayerType layer_type_ = LayerType::kLayerUnknown;
  std::string layer_name_ = "unknown";
}

class Layer : public BaseLayer {
 public:
  virtual Status forward(const tensor::Tensor& input1, const tensor::Tensor& output1) override;

  virtual Status forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                         const tensor::Tensor& output1) override;

  virtual Status forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                         const tensor::Tensor& input3, const tensor::Tensor& output1) override;

  virtual Status forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                         const tensor::Tensor& input3, const tensor::Tensor& input4,
                         const tensor::Tensor& output1) override;

  virtual Status forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                         const tensor::Tensor& input3, const tensor::Tensor& input4,
                         const tensor::Tensor& input5, const tensor::Tensor& output1) override;

  virtual Status to_dev(DeviceType dev_type) override;

 protected:
  virtual Status forward() override;

  virtual void set_input(int32_t idx, const tensor::Tensor& input) override;

  virtual void set_output(int32_t idx, const tensor::Tensor& output) override;

 private:
  std::vector<tensor::Tensor> inputs_;
  std::vector<tensor::Tensor> outputs_;
};

class LayerWithParam : Layer {
 private:
  std::vector<tensor::Tensor> weights_;
};

}  // namespace ginfer::op