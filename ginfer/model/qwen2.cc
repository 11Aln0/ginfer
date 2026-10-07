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
  loadWeights(*m, config);

  return m;
}

// model

Qwen2Model::Qwen2Model(Qwen2Config config, common::DeviceType dev_type)
    : LlamaArchModel(config, dev_type), rotary_emb(dev_type, config.rope_theta) {}

}  // namespace ginfer::model