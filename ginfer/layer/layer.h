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
using ginfer::error::Status;
using ginfer::tensor::Tensor;
using ginfer::tensor::TensorRef;

class BaseLayer {
 public:
  explicit BaseLayer(DeviceType dev_type, std::string layer_name);

  DeviceType getDeviceType() const;

  virtual Status forward(const std::vector<TensorRef>& inputs, TensorRef output) = 0;

  virtual Status toDevice(DeviceType dev_type);

 private:
  DeviceType dev_type_ = DeviceType::kDeviceUnknown;
  std::string layer_name_ = "unknown";
};

class Layer : public BaseLayer {
  using BaseLayer::BaseLayer;
};

class LayerWithParam : public Layer {
  using Layer::Layer;

 public:
  virtual Status toDevice(DeviceType dev_type) override;

  virtual std::vector<TensorRef> getWeights() = 0;
};

class LinearLayer : public LayerWithParam {
 public:
  LinearLayer(DeviceType dev_type, std::string layer_name);

  virtual Status forward(const std::vector<TensorRef>& inputs, TensorRef output) override;

  virtual std::vector<TensorRef> getWeights() override;

  void setWeight(const TensorRef& weight);
  void setBias(const TensorRef& bias);

 private:
  op::MatmulOp mm_op_;
  TensorRef weight_;  // [out_features, in_features]
  TensorRef bias_;    // [out_features]
};

class RMSNormLayer : public LayerWithParam {
 public:
  RMSNormLayer(DeviceType dev_type, std::string layer_name, float epsilon);

  virtual Status forward(const std::vector<TensorRef>& inputs, TensorRef output) override;

  void setWeight(const TensorRef& gamma);

  virtual std::vector<TensorRef> getWeights() override;

 private:
  op::RMSNormOp norm_op_;
  TensorRef gamma_;
};

class EmbeddingLayer : public LayerWithParam {
 public:
  EmbeddingLayer(DeviceType dev_type, std::string layer_name);

  virtual Status forward(const std::vector<TensorRef>& inputs, TensorRef output) override;

  virtual std::vector<TensorRef> getWeights() override;

  void setWeight(const TensorRef& weight);

 private:
  TensorRef weight_;  // [vocab_size, embedding_dim]
  op::EmbeddingOp embed_op_;
};

}  // namespace ginfer::layer