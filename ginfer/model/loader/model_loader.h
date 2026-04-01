#pragma once

#include <memory>
#include <nlohmann/json.hpp>
#include <tuple>
#include "ginfer/common/errors.h"
#include "ginfer/core/layer/layer.h"
#include "ginfer/core/layer/transformer/layer.h"
#include "ginfer/core/memory/allocator_factory.h"
#include "ginfer/core/op/op.h"
#include "ginfer/core/tensor/tensor.h"
#include "ginfer/model/loader/safetensor_loader.h"
#include "ginfer/model/model.h"

namespace ginfer::model {

class ModelLoader {
 public:
  explicit ModelLoader(std::string model_path);

  virtual std::unique_ptr<Model> load() = 0;

  ModelConfig getModelConfig();

 protected:
  using AttentionWeight = core::layer::transformer::AttentionLayer::Weight;
  using EncoderWeight = core::layer::transformer::EncoderLayer::Weight;
  using FeedForwardWeight = core::layer::transformer::FeedForwardLayer::Weight;

 protected:
  nlohmann::json loadConfigJSON();
  core::tensor::DataType parseDataType(const std::string& dtype_str) const;
  void loadModelConfig(ModelConfig& config, const nlohmann::json& json);

  AttentionWeight loadAttentionWeight(
      const std::string& prefix, bool q_bias, bool k_bias, bool v_bias, bool o_bias);
  FeedForwardWeight loadFeedForwardWeight(const std::string& prefix);
  virtual EncoderWeight loadEncoderLayerWeight(int layer_idx);

 protected:
  std::string model_path_;
  SafeTensorLoader weight_loader;
};

class LlamaArchModelLoader : public ModelLoader {
 public:
  using ModelLoader::ModelLoader;

 protected:
  void loadLlamaArchModelConfig(LlamaArchModelConfig& config, const nlohmann::json& json);
  EncoderWeight loadEncoderLayerWeight(int layer_idx) override;
  virtual std::tuple<bool, bool, bool, bool> getAttentionBiasConfig() const = 0;
};

}  // namespace ginfer::model