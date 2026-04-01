#include "ginfer/model/llama3.h"

namespace ginfer::model {

Llama3ModelLoader::Llama3ModelLoader(std::string model_path)
    : LlamaArchModelLoader(std::move(model_path)) {}

std::tuple<bool, bool, bool, bool> Llama3ModelLoader::getAttentionBiasConfig() const {
  return {false, false, false, false};
}

Llama3Config Llama3ModelLoader::loadConfig() {
  Llama3Config config;

  auto json = loadConfigJSON();
  loadLlamaArchModelConfig(config, json);

  auto rope_scaling_json = json.value("rope_scaling", nlohmann::json::object());
  config.rope_scaling.factor = rope_scaling_json.value("factor", 1.0f);
  config.rope_scaling.high_freq_factor = rope_scaling_json.value("high_freq_factor", 1.0f);
  config.rope_scaling.low_freq_factor = rope_scaling_json.value("low_freq_factor", 1.0f);
  config.rope_scaling.old_ctx_len = json.value("original_max_position_embeddings", 8192);

  return config;
}

std::unique_ptr<Model> Llama3ModelLoader::load() {
  Llama3Config config = loadConfig();
  auto m = std::make_unique<Llama3Model>(config);

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

}  // namespace ginfer::model