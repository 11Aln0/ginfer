#include <gtest/gtest.h>
#include <string>
#include "ginfer/model/qwen2.h"

namespace {

using ginfer::model::Qwen2Config;
using ginfer::model::Qwen2Model;

// Expose the layers of the model so the test can inspect their names.
class NamedQwen2Model : public Qwen2Model {
 public:
  using Qwen2Model::Qwen2Model;
  using ginfer::model::LlamaArchModel::embed_tokens;
  using ginfer::model::LlamaArchModel::encoder_layers;
  using ginfer::model::LlamaArchModel::final_rmsnorm;
  using ginfer::model::LlamaArchModel::lm_head;
};

Qwen2Config makeConfig(int nlayer) {
  Qwen2Config config;
  config.dtype = ginfer::core::tensor::DataType::kDataTypeBFloat16;
  config.nlayer = nlayer;
  config.vocab_size = 32;
  config.max_position_embeddings = 64;
  config.num_heads = 4;
  config.num_kv_heads = 2;
  config.head_dim = 8;
  config.hidden_size = 32;
  config.intermediate_size = 64;
  config.rms_norm_eps = 1e-6f;
  config.rope_theta = 10000.0f;
  config.tie_word_embeddings = false;
  return config;
}

}  // namespace

// Layer names are full HF-style paths; the loader derives checkpoint keys from them
// ("model." + name + ".weight"), so they must not drift from the HF module names.
TEST(LayerNameTest, modelLevelLayers) {
  NamedQwen2Model model(makeConfig(2));

  EXPECT_EQ(model.embed_tokens.name(), "embed_tokens");
  EXPECT_EQ(model.final_rmsnorm.name(), "norm");
  EXPECT_EQ(model.lm_head.name(), "lm_head");
  ASSERT_EQ(model.encoder_layers.size(), 2u);
  EXPECT_EQ(model.encoder_layers[0].name(), "layers.0");
  EXPECT_EQ(model.encoder_layers[1].name(), "layers.1");
}

TEST(LayerNameTest, encoderLayerChildren) {
  NamedQwen2Model model(makeConfig(2));
  auto& layer = model.encoder_layers[1];

  EXPECT_EQ(layer.getAttnNormLayer().name(), "layers.1.input_layernorm");
  EXPECT_EQ(layer.getMLPNormLayer().name(), "layers.1.post_attention_layernorm");

  auto& attn = layer.getAttentionLayer();
  EXPECT_EQ(attn.name(), "layers.1.self_attn");
  EXPECT_EQ(attn.qProj().name(), "layers.1.self_attn.q_proj");
  EXPECT_EQ(attn.kProj().name(), "layers.1.self_attn.k_proj");
  EXPECT_EQ(attn.vProj().name(), "layers.1.self_attn.v_proj");
  EXPECT_EQ(attn.oProj().name(), "layers.1.self_attn.o_proj");

  auto& mlp = layer.getFeedForwardLayer();
  EXPECT_EQ(mlp.name(), "layers.1.mlp");
  EXPECT_EQ(mlp.gateProj().name(), "layers.1.mlp.gate_proj");
  EXPECT_EQ(mlp.upProj().name(), "layers.1.mlp.up_proj");
  EXPECT_EQ(mlp.downProj().name(), "layers.1.mlp.down_proj");
}
