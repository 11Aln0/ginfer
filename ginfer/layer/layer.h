#pragma once

#include <memory>
#include <string>
#include <vector>

#include "ginfer/common/device.h"
#include "ginfer/common/errors.h"
#include "ginfer/tensor/tensor.h"

namespace ginfer::op {

using ginfer::common::DeviceType;
using ginfer::error::Status;
using ginfer::tensor::Tensor;

enum class LayerType : uint8_t {
  kLayerUnknown = 0,
  kLayerLinear = 1,
  kLayerEncode = 2,
  kLayerEmbedding = 3,
  kLayerRMSNorm = 4,
  kLayerMatmul = 5,
  kLayerROPE = 6,
  kLayerMHA = 7,
  kLayerGQA = 8,
  kLayerSoftmax = 9,
  kLayerAdd = 10,
  kLayerSwiGLU = 11,
  kLayerArgmax = 12,
};

class BaseLayer {
 public:
  explicit BaseLayer(DeviceType dev_type, LayerType layer_type, std::string layer_name);

  LayerType layerType() const;

  DeviceType getDeviceType() const;

  virtual Status forward(const std::vector<const Tensor*>& inputs, Tensor* output) = 0;

  virtual Status toDevice(DeviceType dev_type);

 protected:
  // virtual Status forward() = 0;

  // virtual void set_input(int32_t idx, const tensor::Tensor& input) = 0;

  // virtual void set_output(int32_t idx, const tensor::Tensor& output) = 0;

 private:
  DeviceType dev_type_ = DeviceType::kDeviceUnknown;
  LayerType layer_type_ = LayerType::kLayerUnknown;
  std::string layer_name_ = "unknown";
};

class Layer : public BaseLayer {
  using BaseLayer::BaseLayer;
  //  public:
  // virtual Status forward(std::vector<const Tensor*> inputs, Tensor* output) override;

  // virtual Status toDevice(DeviceType dev_type) override;

  //  protected:
  //   virtual Status forward() override;

  // virtual void set_input(int32_t idx, const tensor::Tensor& input) override;
  // virtual void set_output(int32_t idx, const tensor::Tensor& output) override;

  //  private:
  //   std::vector<tensor::Tensor> inputs_;
  //   std::vector<tensor::Tensor> outputs_;
};

class LayerWithParam : public Layer {
  using Layer::Layer;

 public:
  void resetWeightSize(size_t size);

  void setWeight(int32_t idx, std::shared_ptr<Tensor> weight);  // TODO: avoid copy

  virtual Status toDevice(DeviceType dev_type) override;

  std::shared_ptr<Tensor> getWeight(int32_t idx);

  std::vector<std::shared_ptr<Tensor>> getWeights();

 private:
  std::vector<std::shared_ptr<Tensor>> weights_;
};

class MatmulLayer : public Layer {
 public:
  MatmulLayer(DeviceType dev_type, std::string layer_name);

  virtual Status forward(const std::vector<const Tensor*>& inputs, Tensor* output) override;
};

class AddLayer : public Layer {
 public:
  AddLayer(DeviceType dev_type, std::string layer_name);

  virtual Status forward(const std::vector<const Tensor*>& inputs, Tensor* output) override;

  //  private:
  //   Status checkParams(const std::vector<const Tensor*>& inputs, const Tensor* output);
};

class RMSNormLayer : public LayerWithParam {
 public:
  RMSNormLayer(DeviceType dev_type, std::string layer_name, float epsilon);

  virtual Status forward(const std::vector<const Tensor*>& inputs, Tensor* output) override;

 private:
  float epsilon_;
};

class GQALayer : public Layer {
 public:
  GQALayer(DeviceType dev_type, std::string layer_name);

  virtual Status forward(const std::vector<const Tensor*>& inputs, Tensor* output) override;

  void setSeqLen(int seq_len);

 private:
  int seq_len_;
};

class ArgmaxLayer : public Layer {
 public:
  ArgmaxLayer(DeviceType dev_type, std::string layer_name);

  virtual Status forward(const std::vector<const Tensor*>& inputs, Tensor* output) override;
};

class EmbeddingLayer : public LayerWithParam {
 public:
  EmbeddingLayer(DeviceType dev_type, std::string layer_name);

  virtual Status forward(const std::vector<const Tensor*>& inputs, Tensor* output) override;
};

class ROPELayer : public Layer {
 public:
  ROPELayer(DeviceType dev_type, std::string layer_name, int head_dim, int max_seq_len, float rope_theta = 10000.0f);

  virtual Status forward(const std::vector<const Tensor*>& inputs, Tensor* output) override;

  void updateCache(int start_pos, int end_pos);

 private:
  int head_dim_;
  int max_seq_len_;
  float rope_theta_;
  std::shared_ptr<Tensor> sin_cache_;
  std::shared_ptr<Tensor> cos_cache_;
};

class SwiGLULayer : public Layer {
 public:
  SwiGLULayer(DeviceType dev_type, std::string layer_name);

  virtual Status forward(const std::vector<const Tensor*>& inputs, Tensor* output) override;
};

}  // namespace ginfer::op