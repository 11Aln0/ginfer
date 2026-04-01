#include "ginfer/model/qwen2.h"
#include <fstream>
#include <nlohmann/json.hpp>
#include "ginfer/common/errors.h"
#include "ginfer/core/layer/layer.h"
#include "ginfer/core/layer/transformer/layer.h"
#include "ginfer/model/loader/safetensor_loader.h"

namespace ginfer::model {

// loader
Qwen2ModelLoader::Qwen2ModelLoader(std::string model_path)
    : LlamaArchModelLoader(std::move(model_path)) {}

std::tuple<bool, bool, bool, bool> Qwen2ModelLoader::getAttentionBiasConfig() const {
  return {true, true, true, false};
}

Qwen2Config Qwen2ModelLoader::loadConfig() {
  Qwen2Config config;

  auto json = loadConfigJSON();
  loadLlamaArchModelConfig(config, json);

  return config;
}

std::unique_ptr<Model> Qwen2ModelLoader::load() {
  Qwen2Config config = loadConfig();
  auto m = std::make_unique<Qwen2Model>(config);

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

// model

Qwen2Model::Qwen2Model(Qwen2Config config, common::DeviceType dev_type)
    : LlamaArchModel(config, dev_type), rotary_emb(dev_type, config.rope_theta) {}

}  // namespace ginfer::model