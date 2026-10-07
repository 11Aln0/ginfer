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
  nlohmann::json loadConfigJSON();
  core::tensor::DataType parseDataType(const std::string& dtype_str) const;
  void loadModelConfig(ModelConfig& config, const nlohmann::json& json);

  // Checkpoint (HF safetensors) key of a layer parameter, derived from the layer's full path:
  // everything lives under "model." except lm_head, e.g. "layers.0.self_attn.q_proj" + "weight"
  // -> "model.layers.0.self_attn.q_proj.weight".
  std::string ckptKey(const std::string& layer_name, const std::string& suffix) const;

  // Leaf layer loaders: the checkpoint keys come from layer.name().
  void loadLinear(core::layer::LinearLayer& layer, bool has_bias);
  void loadRMSNorm(core::layer::RMSNormLayer& layer);
  // Returns the loaded weight so that callers can share it (tied embeddings).
  core::tensor::TensorRef loadEmbedding(core::layer::EmbeddingLayer& layer);

 protected:
  std::string model_path_;
  SafeTensorLoader weight_loader;
};

class LlamaArchModelLoader : public ModelLoader {
 public:
  using ModelLoader::ModelLoader;

 protected:
  void loadLlamaArchModelConfig(LlamaArchModelConfig& config, const nlohmann::json& json);

  // Fill all weights of `model` from the safetensors file that was loaded into weight_loader.
  void loadWeights(LlamaArchModel& model, const LlamaArchModelConfig& config);

  void loadEncoderLayer(core::layer::transformer::EncoderLayer& layer);
  void loadAttention(core::layer::transformer::AttentionLayer& layer);
  void loadFeedForward(core::layer::transformer::FeedForwardLayer& layer);

  // {q_bias, k_bias, v_bias, o_bias}
  virtual std::tuple<bool, bool, bool, bool> getAttentionBiasConfig() const = 0;
};

}  // namespace ginfer::model