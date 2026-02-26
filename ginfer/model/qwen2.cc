#include "ginfer/model/qwen2.h"
#include <fstream>
#include <nlohmann/json.hpp>
#include "ginfer/common/errors.h"
#include "ginfer/layer/layer.h"
#include "ginfer/layer/transformer/layer.h"
#include "ginfer/model/loader/safetensor_loader.h"

namespace ginfer::model {

// loader
Qwen2ModelLoader::Qwen2ModelLoader(std::string model_path) : ModelLoader(std::move(model_path)) {}

Qwen2Config Qwen2ModelLoader::loadConfig() {
  Qwen2Config config;

  auto json = loadConfigJSON();
  config.dtype = parseDataType(json.value("torch_dtype", "float16"));

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
  if (auto it = json.find("eos_token_id"); it != json.end() && it->is_array()) {
    config.eos_token_ids = it->get<std::vector<int32_t>>();
  } else {
    config.eos_token_ids = {json.value("eos_token_id", static_cast<int32_t>(151645))};
  }
  config.tie_word_embeddings = json.value("tie_word_embeddings", false);

  return config;
}

std::shared_ptr<Model> Qwen2ModelLoader::load() {
  Qwen2Config config = loadConfig();
  auto m = std::make_shared<Qwen2Model>(config);

  weight_loader.load(model_path_ + "/model.safetensors");  // TODO multi-part safetensors

  auto embed_weight = weight_loader.getTensor("model.embed_tokens.weight");
  m->embed_tokens.setWeight(embed_weight);
  for (int i = 0; i < config.nlayer; i++) {
    auto& encoder = m->encoder_layers[i];
    encoder.setWeight(loadEncoderLayerWeight(i));
  }

  m->final_rmsnorm.setWeight(weight_loader.getTensor("model.norm.weight"));
  if (config.tie_word_embeddings) {
    m->lm_head.setWeight(embed_weight);
  } else {
    m->lm_head.setWeight(weight_loader.getTensor("lm_head.weight"));
  }

  return m;
}

ModelLoader::EncoderWeight Qwen2ModelLoader::loadEncoderLayerWeight(int layer_idx) {
  std::string prefix = "model.layers." + std::to_string(layer_idx);
  auto attn = loadAttentionWeight(prefix, true, true, true, false);
  auto mlp = loadFeedForwardWeight(prefix);

  EncoderWeight w = {
      .attn = attn,
      .mlp = mlp,
      .attn_norm = weight_loader.getTensor(prefix + ".input_layernorm.weight"),
      .mlp_norm = weight_loader.getTensor(prefix + ".post_attention_layernorm.weight"),
  };

  return w;
}

// model

Qwen2Model::Qwen2Model(Qwen2Config config, common::DeviceType dev_type)
    : LlamaArchModel(config, dev_type), rotary_emb(dev_type, config.rope_theta) {}

}  // namespace ginfer::model