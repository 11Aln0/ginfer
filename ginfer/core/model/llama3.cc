#include "ginfer/core/model/llama3.h"

namespace ginfer::core::model {

Llama3ModelLoader::Llama3ModelLoader(std::string model_path) : ModelLoader(std::move(model_path)) {}

Llama3Config Llama3ModelLoader::loadConfig() {
  Llama3Config config;

  auto json = loadConfigJSON();
  config.dtype = parseDataType(json.value("torch_dtype", "float16"));

  config.nlayer = json.at("num_hidden_layers").get<int>();
  config.hidden_size = json.at("hidden_size").get<int>();
  config.num_heads = json.at("num_attention_heads").get<int>();
  config.num_kv_heads = json.at("num_key_value_heads").get<int>();
  config.head_dim = json.value("head_dim", config.hidden_size / config.num_heads);
  config.intermediate_size = json.at("intermediate_size").get<int>();
  config.vocab_size = json.at("vocab_size").get<int>();
  config.max_seq_len = 4096;  // temporary fix for llama3 models with longer context
  config.rms_norm_eps = json.value("rms_norm_eps", 1e-6f);
  config.rope_theta = json.value("rope_theta", 10000.0f);
  if (auto it = json.find("eos_token_id"); it != json.end() && it->is_array()) {
    config.eos_token_ids = it->get<std::vector<int32_t>>();
  } else {
    config.eos_token_ids = {
        json.value("eos_token_id", static_cast<int32_t>(config.vocab_size - 1))};
  }
  config.tie_word_embeddings = json.value("tie_word_embeddings", false);

  auto rope_scaling_json = json.value("rope_scaling", nlohmann::json::object());
  config.rope_scaling.factor = rope_scaling_json.value("factor", 1.0f);
  config.rope_scaling.high_freq_factor = rope_scaling_json.value("high_freq_factor", 1.0f);
  config.rope_scaling.low_freq_factor = rope_scaling_json.value("low_freq_factor", 1.0f);
  config.rope_scaling.old_ctx_len = json.value("original_max_position_embeddings", 8192);

  return config;
}

ModelLoader::EncoderWeight Llama3ModelLoader::loadEncoderLayerWeight(int layer_idx) {
  std::string prefix = "model.layers." + std::to_string(layer_idx);
  auto attn = loadAttentionWeight(prefix, false, false, false, false);
  auto mlp = loadFeedForwardWeight(prefix);

  EncoderWeight w = {
      .attn = attn,
      .mlp = mlp,
      .attn_norm = weight_loader.getTensor(prefix + ".input_layernorm.weight"),
      .mlp_norm = weight_loader.getTensor(prefix + ".post_attention_layernorm.weight"),
  };

  return w;
}

std::shared_ptr<Model> Llama3ModelLoader::load() {
  Llama3Config config = loadConfig();
  auto m = std::make_shared<Llama3Model>(config);

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

Llama3Model::Llama3Model(Llama3Config cfg, common::DeviceType dev_type)
    : LlamaArchModel(cfg, dev_type),
      rotary_emb(dev_type,
                 cfg.rope_theta,
                 cfg.rope_scaling.factor,
                 cfg.rope_scaling.high_freq_factor,
                 cfg.rope_scaling.low_freq_factor,
                 cfg.rope_scaling.old_ctx_len) {}

}  // namespace ginfer::core::model