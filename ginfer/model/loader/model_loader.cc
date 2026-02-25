
#include "ginfer/model/loader/model_loader.h"
#include <fstream>
#include "ginfer/common/errors.h"
#include "ginfer/layer/layer.h"
#include "ginfer/layer/transformer/layer.h"

namespace ginfer::model {

ModelLoader::ModelLoader(std::string model_path) : model_path_(std::move(model_path)) {}

nlohmann::json ModelLoader::loadConfigJSON() {
  std::ifstream f(model_path_ + "/config.json");
  CHECK_THROW(f.is_open(), "Failed to open config.json at ", model_path_);
  return nlohmann::json::parse(f);
}

tensor::DataType ModelLoader::parseDataType(const std::string& dtype_str) const {
  using tensor::DataType;
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

ModelLoader::AttentionWeight ModelLoader::loadAttentionWeight(const std::string& prefix) {
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

}  // namespace ginfer::model