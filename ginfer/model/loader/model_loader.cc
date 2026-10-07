
#include "ginfer/model/loader/model_loader.h"
#include <fstream>
#include "ginfer/common/errors.h"
#include "ginfer/core/layer/layer.h"
#include "ginfer/core/layer/transformer/layer.h"

namespace ginfer::model {

ModelLoader::ModelLoader(std::string model_path) : model_path_(std::move(model_path)) {}

nlohmann::json ModelLoader::loadConfigJSON() {
  std::ifstream f(model_path_ + "/config.json");
  CHECK_THROW(f.is_open(), "Failed to open config.json at ", model_path_);
  return nlohmann::json::parse(f);
}

ModelConfig ModelLoader::getModelConfig() {
  ModelConfig config;
  auto json = loadConfigJSON();
  loadModelConfig(config, json);
  return config;
}

core::tensor::DataType ModelLoader::parseDataType(const std::string& dtype_str) const {
  using core::tensor::DataType;
  if (dtype_str == "bfloat16") {
    return DataType::kDataTypeBFloat16;
  } else if (dtype_str == "float16") {
    return DataType::kDataTypeFloat16;
  } else if (dtype_str == "float32") {
    return DataType::kDataTypeFloat32;
  } else {
    CHECK_THROW(false, "Unsupported data type: {}", dtype_str);
  }
}

void ModelLoader::loadModelConfig(ModelConfig& config, const nlohmann::json& json) {
  config.dtype = parseDataType(json.value("torch_dtype", "float16"));
  config.nlayer = json.at("num_hidden_layers").get<int>();
  config.vocab_size = json.at("vocab_size").get<int>();
  config.max_position_embeddings = json.at("max_position_embeddings").get<int>();
  config.num_heads = json.at("num_attention_heads").get<int>();
  config.num_kv_heads = json.at("num_key_value_heads").get<int>();
  config.head_dim = json.value("head_dim", 0);
  if (auto it = json.find("eos_token_id"); it != json.end() && it->is_array()) {
    config.eos_token_ids = it->get<std::vector<int32_t>>();
  } else {
    config.eos_token_ids = {
        json.value("eos_token_id", static_cast<int32_t>(config.vocab_size - 1))};
  }
}

std::string ModelLoader::ckptKey(const std::string& layer_name,
                                 const std::string& suffix) const {
  // HF keeps everything except lm_head under the `model` submodule.
  const std::string prefix = layer_name == "lm_head" ? "" : "model.";
  return prefix + layer_name + "." + suffix;
}

void ModelLoader::loadLinear(core::layer::LinearLayer& layer, bool has_bias) {
  layer.setWeight(weight_loader.getTensor(ckptKey(layer.name(), "weight")));
  if (has_bias) {
    layer.setBias(weight_loader.getTensor(ckptKey(layer.name(), "bias")));
  }
}

void ModelLoader::loadRMSNorm(core::layer::RMSNormLayer& layer) {
  layer.setWeight(weight_loader.getTensor(ckptKey(layer.name(), "weight")));
}

core::tensor::TensorRef ModelLoader::loadEmbedding(core::layer::EmbeddingLayer& layer) {
  auto weight = weight_loader.getTensor(ckptKey(layer.name(), "weight"));
  layer.setWeight(weight);
  return weight;
}

void LlamaArchModelLoader::loadLlamaArchModelConfig(LlamaArchModelConfig& config,
                                                    const nlohmann::json& json) {
  loadModelConfig(config, json);
  config.hidden_size = json.at("hidden_size").get<int>();
  config.intermediate_size = json.at("intermediate_size").get<int>();
  config.rms_norm_eps = json.value("rms_norm_eps", 1e-6f);
  config.rope_theta = json.value("rope_theta", 10000.0f);
  config.tie_word_embeddings = json.value("tie_word_embeddings", false);
  if (config.head_dim == 0) {
    config.head_dim = config.hidden_size / config.num_heads;
  }
}

void LlamaArchModelLoader::loadAttention(core::layer::transformer::AttentionLayer& layer) {
  auto [q_bias, k_bias, v_bias, o_bias] = getAttentionBiasConfig();
  loadLinear(layer.qProj(), q_bias);
  loadLinear(layer.kProj(), k_bias);
  loadLinear(layer.vProj(), v_bias);
  loadLinear(layer.oProj(), o_bias);
}

void LlamaArchModelLoader::loadFeedForward(core::layer::transformer::FeedForwardLayer& layer) {
  loadLinear(layer.gateProj(), false);
  loadLinear(layer.upProj(), false);
  loadLinear(layer.downProj(), false);
}

void LlamaArchModelLoader::loadEncoderLayer(core::layer::transformer::EncoderLayer& layer) {
  loadRMSNorm(layer.getAttnNormLayer());
  loadAttention(layer.getAttentionLayer());
  loadRMSNorm(layer.getMLPNormLayer());
  loadFeedForward(layer.getFeedForwardLayer());
}

void LlamaArchModelLoader::loadWeights(LlamaArchModel& model, const LlamaArchModelConfig& config) {
  auto embed_weight = loadEmbedding(model.embed_tokens);
  for (auto& encoder : model.encoder_layers) {
    loadEncoderLayer(encoder);
  }
  loadRMSNorm(model.final_rmsnorm);
  if (config.tie_word_embeddings) {
    model.lm_head.setWeight(embed_weight);
  } else {
    model.lm_head.setWeight(weight_loader.getTensor(ckptKey(model.lm_head.name(), "weight")));
  }
}

}  // namespace ginfer::model
