#include <fstream>
#include <nlohmann/json.hpp>
#include "ginfer/common/errors.h"
#include "ginfer/layer/layer.h"
#include "ginfer/layer/transformer/layer.h"
#include "ginfer/model/qwen2.h"
#include "ginfer/model/safetensor_loader.h"

namespace ginfer::model {

Qwen2ModelLoader::Qwen2ModelLoader(std::string model_path) : model_path_(std::move(model_path)) {}

Qwen2Config Qwen2ModelLoader::loadConfig() {
  Qwen2Config config;
  std::ifstream f(model_path_ + "/config.json");
  CHECK_THROW(f.is_open(), "Failed to open config.json at ", model_path_);
  auto json = nlohmann::json::parse(f);

  // dtype
  std::string torch_dtype = json.value("torch_dtype", "float16");
  if (torch_dtype == "bfloat16") {
    config.dtype = tensor::DataType::kDataTypeBFloat16;
  } else if (torch_dtype == "float16") {
    config.dtype = tensor::DataType::kDataTypeFloat16;
  } else if (torch_dtype == "float32") {
    config.dtype = tensor::DataType::kDataTypeFloat32;
  } else {
    CHECK_THROW(false, "Unsupported torch_dtype: ", torch_dtype);
  }

  config.nlayer = json.at("num_hidden_layers").get<int>();
  config.hidden_size = json.at("hidden_size").get<int>();
  config.num_heads = json.at("num_attention_heads").get<int>();
  config.num_kv_heads = json.at("num_key_value_heads").get<int>();
  config.head_dim = json.value("head_dim", config.hidden_size / config.num_heads);
  config.intermediate_size = json.at("intermediate_size").get<int>();
  config.vocab_size = json.at("vocab_size").get<int>();
  // config.max_seq_len = json.at("max_position_embeddings").get<size_t>();
  config.max_seq_len = 4096;  // TODO temporary fix for qwen2 models with longer context
  config.rms_norm_eps = json.value("rms_norm_eps", 1e-6f);
  config.rope_theta = json.value("rope_theta", 10000.0f);
  config.eos_token_id = json.value("eos_token_id", static_cast<int64_t>(151645));

  return config;
}

Qwen2ModelLoader::AttentionWeight Qwen2ModelLoader::loadAttentionWeight(const std::string& prefix) {
  std::string attn_prefix = prefix + ".self_attn";
  AttentionWeight w = {
      .q_w = weight_loader.getTensor(attn_prefix + ".q_proj.weight"),
      .q_b = weight_loader.getTensor(attn_prefix + ".q_proj.bias"),
      .k_w = weight_loader.getTensor(attn_prefix + ".k_proj.weight"),
      .k_b = weight_loader.getTensor(attn_prefix + ".k_proj.bias"),
      .v_w = weight_loader.getTensor(attn_prefix + ".v_proj.weight"),
      .v_b = weight_loader.getTensor(attn_prefix + ".v_proj.bias"),
      .o_w = weight_loader.getTensor(attn_prefix + ".o_proj.weight"),
  };
  return w;
}

Qwen2ModelLoader::FeedForwardWeight Qwen2ModelLoader::loadFeedForwardWeight(const std::string& prefix) {
  std::string mlp_prefix = prefix + ".mlp";
  FeedForwardWeight w = {
      .gate_w = weight_loader.getTensor(mlp_prefix + ".gate_proj.weight"),
      .up_w = weight_loader.getTensor(mlp_prefix + ".up_proj.weight"),
      .down_w = weight_loader.getTensor(mlp_prefix + ".down_proj.weight"),
  };
  return w;
}

Qwen2ModelLoader::EncoderWeight Qwen2ModelLoader::loadEncoderLayerWeight(int layer_idx) {
  std::string prefix = "model.layers." + std::to_string(layer_idx);
  auto attn = loadAttentionWeight(prefix);
  auto mlp = loadFeedForwardWeight(prefix);

  EncoderWeight w = {
      .attn = attn,
      .mlp = mlp,
      .attn_norm = weight_loader.getTensor(prefix + ".input_layernorm.weight"),
      .mlp_norm = weight_loader.getTensor(prefix + ".post_attention_layernorm.weight"),
  };

  return w;
}

std::shared_ptr<Qwen2Model> Qwen2ModelLoader::load() {
  Qwen2Config config = loadConfig();
  auto m = std::make_shared<Qwen2Model>(config);

  SafeTensorLoader weight_loader;
  weight_loader.load(model_path_ + "/model.safetensors");  // TODO multi-part safetensors

  m->embed_tokens.setWeight(weight_loader.getTensor("model.embed_tokens.weight"));
  for (int i = 0; i < m->config_.nlayer; i++) {
    auto& encoder = m->encoder_layers[i];
    encoder.setWeight(loadEncoderLayerWeight(i));
  }

  m->final_rmsnorm.setWeight(weight_loader.getTensor("model.norm.weight"));
  m->lm_head.setWeight(weight_loader.getTensor("lm_head.weight"));

  return m;
}

}  // namespace ginfer::model