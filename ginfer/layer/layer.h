#pragma once

#include <glog/logging.h>
#include <memory>
#include <string>
#include <vector>

#include "ginfer/common/device.h"
#include "ginfer/common/errors.h"
#include "ginfer/op/op.h"
#include "ginfer/tensor/tensor.h"

namespace ginfer::layer {

using ginfer::common::DeviceType;
using ginfer::tensor::Tensor;
using ginfer::tensor::TensorRef;

class BaseLayer {
 public:
  explicit BaseLayer(DeviceType dev_type, std::string layer_name);

  DeviceType getDeviceType() const;

  virtual Result<void, std::string> forward(const std::vector<TensorRef>& inputs, TensorRef output) = 0;

  virtual Result<void, std::string> toDevice(DeviceType dev_type);

 private:
  DeviceType dev_type_ = DeviceType::kDeviceUnknown;
  std::string layer_name_ = "unknown";
};

class Layer : public BaseLayer {
  using BaseLayer::BaseLayer;
};

// class LayerWithParam : public Layer {
//   using Layer::Layer;

//  public:
//   virtual Result<void, std::string> toDevice(DeviceType dev_type) override;

//   virtual std::vector<TensorRef> getWeights() = 0;
// };

class LinearLayer : public Layer {
 public:
  LinearLayer(DeviceType dev_type, std::string layer_name);

  virtual Result<void, std::string> forward(const std::vector<TensorRef>& inputs, TensorRef output) override;

  virtual Result<void, std::string> toDevice(DeviceType dev_type) override;

  void setWeight(const TensorRef& weight);
  void setBias(const TensorRef& bias);

 private:
  op::MatmulOp mm_op_;
  TensorRef weight_;  // [out_features, in_features]
  TensorRef bias_;    // [out_features]
};

class RMSNormLayer : public Layer {
 public:
  RMSNormLayer(DeviceType dev_type, std::string layer_name, float epsilon);

  virtual Result<void, std::string> forward(const std::vector<TensorRef>& inputs, TensorRef output) override;

  void setWeight(const TensorRef& gamma);

  virtual Result<void, std::string> toDevice(DeviceType dev_type) override;

 private:
  op::RMSNormOp norm_op_;
  TensorRef gamma_;
};

class EmbeddingLayer : public Layer {
 public:
  EmbeddingLayer(DeviceType dev_type, std::string layer_name);

  virtual Result<void, std::string> forward(const std::vector<TensorRef>& inputs, TensorRef output) override;

  virtual Result<void, std::string> toDevice(DeviceType dev_type) override;

  void setWeight(const TensorRef& weight);

 private:
  TensorRef weight_;  // [vocab_size, embedding_dim]
  op::EmbeddingOp embed_op_;
};

}  // namespace ginfer::layer
