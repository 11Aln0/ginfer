
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

ModelLoader::AttentionWeight ModelLoader::loadAttentionWeight(
    const std::string& prefix, bool q_bias, bool k_bias, bool v_bias, bool o_bias) {
  std::string attn_prefix = prefix + ".self_attn";
  AttentionWeight w = {
      .q_w = weight_loader.getTensor(attn_prefix + ".q_proj.weight"),
      .k_w = weight_loader.getTensor(attn_prefix + ".k_proj.weight"),
      .v_w = weight_loader.getTensor(attn_prefix + ".v_proj.weight"),
      .o_w = weight_loader.getTensor(attn_prefix + ".o_proj.weight"),
  };
  if (q_bias) w.q_b = weight_loader.getTensor(attn_prefix + ".q_proj.bias");
  if (k_bias) w.k_b = weight_loader.getTensor(attn_prefix + ".k_proj.bias");
  if (v_bias) w.v_b = weight_loader.getTensor(attn_prefix + ".v_proj.bias");
  if (o_bias) w.o_b = weight_loader.getTensor(attn_prefix + ".o_proj.bias");
  return w;
}

ModelLoader::FeedForwardWeight ModelLoader::loadFeedForwardWeight(const std::string& prefix) {
  std::string mlp_prefix = prefix + ".mlp";
  FeedForwardWeight w = {
      .gate_w = weight_loader.getTensor(mlp_prefix + ".gate_proj.weight"),
      .up_w = weight_loader.getTensor(mlp_prefix + ".up_proj.weight"),
      .down_w = weight_loader.getTensor(mlp_prefix + ".down_proj.weight"),
  };
  return w;
}

ModelLoader::EncoderWeight ModelLoader::loadEncoderLayerWeight(int layer_idx) {
  std::string prefix = "model.layers." + std::to_string(layer_idx);
  auto attn = loadAttentionWeight(prefix, true, true, true, true);
  auto mlp = loadFeedForwardWeight(prefix);

  EncoderWeight w = {
      .attn = attn,
      .mlp = mlp,
      .attn_norm = weight_loader.getTensor(prefix + ".input_layernorm.weight"),
      .mlp_norm = weight_loader.getTensor(prefix + ".post_attention_layernorm.weight"),
  };

  return w;
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

ModelLoader::EncoderWeight LlamaArchModelLoader::loadEncoderLayerWeight(int layer_idx) {
  std::string prefix = "model.layers." + std::to_string(layer_idx);
  auto [q_bias, k_bias, v_bias, o_bias] = getAttentionBiasConfig();
  auto attn = loadAttentionWeight(prefix, q_bias, k_bias, v_bias, o_bias);
  auto mlp = loadFeedForwardWeight(prefix);

  EncoderWeight w = {
      .attn = attn,
      .mlp = mlp,
      .attn_norm = weight_loader.getTensor(prefix + ".input_layernorm.weight"),
      .mlp_norm = weight_loader.getTensor(prefix + ".post_attention_layernorm.weight"),
  };

  return w;
}

}  // namespace ginfer::model