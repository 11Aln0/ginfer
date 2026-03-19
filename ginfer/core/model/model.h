#pragma once

#include <memory>
#include <tuple>
#include <vector>
#include "ginfer/common/errors.h"
#include "ginfer/core/layer/layer.h"
#include "ginfer/core/layer/transformer/layer.h"
#include "ginfer/core/memory/allocator_factory.h"
#include "ginfer/core/model/loader/safetensor_loader.h"
#include "ginfer/core/op/op.h"
#include "ginfer/core/tensor/tensor.h"

namespace ginfer::core::model {

using ginfer::core::tensor::Tensor;
using ginfer::core::tensor::TensorRef;

struct ModelConfig {
 public:
  tensor::DataType dtype;

  int nlayer;
  int vocab_size;
  int max_position_embeddings;
  int max_seq_len;

  int num_heads;
  int num_kv_heads;
  int head_dim;

  std::vector<int32_t> eos_token_ids;
};

class Model {
 public:
  Model() = delete;

  explicit Model(ModelConfig config, common::DeviceType dev_type)
      : config_(std::move(config)), dev_type_(dev_type) {}

  virtual Result<int32_t, std::string> predict(const core::InferContext& infer_ctx,
                                               const tensor::TensorRef token_ids,
                                               const tensor::TensorRef positions) = 0;

  virtual Result<void, std::string> toDevice(common::DeviceType dev_type);

  int getVocabSize() const;

  int getNumLayers() const;

  tensor::DataType getDtype() const;

  std::tuple<int, int, int> getAttentionConfig() const;

  bool isEosToken(int32_t token_id) const;

  // K/V Cache Tensor: [num_kvcache_blocks, block_size, num_kv_heads * head_dim]
  virtual void setKVCache(int layer_id, TensorRef& k_cache, TensorRef& v_cache) = 0;

 private:
 private:
  ModelConfig config_;
  common::DeviceType dev_type_;
};

struct LlamaArchModelConfig : public ModelConfig {
  int hidden_size;
  int intermediate_size;

  float rms_norm_eps;
  float rope_theta;

  bool tie_word_embeddings;
};

class LlamaArchModel : public Model {
 public:
  LlamaArchModel(LlamaArchModelConfig config, common::DeviceType dev_type);

  Result<int32_t, std::string> predict(const core::InferContext& infer_ctx,
                                       const tensor::TensorRef token_ids,
                                       const tensor::TensorRef positions) override;

  Result<void, std::string> toDevice(common::DeviceType dev_type) override;

  void setKVCache(int layer_id, TensorRef& k_cache, TensorRef& v_cache) override;

 private:
  struct Intermediates {
    TensorRef embed_out;    // [max_seq_len, hidden_size]
    TensorRef norm_out;     // [max_seq_len, hidden_size]
    TensorRef lm_head_out;  // [max_seq_len, vocab_size]
    TensorRef argmax_out;   // [1]
  };

 private:
  // mem
  Result<void, std::string> lazyAllocIntermediates();
  // Result<void, std::string> lazyAllocKVCache();

  Result<void, std::string> lazyInitPosEmbedding();

  // forward
  Result<void, std::string> forward(const core::InferContext& infer_ctx,
                                    const TensorRef input_ids,
                                    const TensorRef positions,
                                    TensorRef output);

 private:
  LlamaArchModelConfig config_;
  common::DeviceType dev_type_;
  bool intermediates_allocated_ = false;

  Intermediates intermediates_;

 protected:
  TensorRef sin;
  TensorRef cos;
  bool pos_embedding_initialized_ = false;

  virtual op::Op& getRotaryEmbeddingOp() = 0;

 protected:
  op::ArgmaxOp argmax_op;
  layer::EmbeddingLayer embed_tokens;
  std::vector<layer::transformer::EncoderLayer> encoder_layers;
  layer::RMSNormLayer final_rmsnorm;
  layer::LinearLayer lm_head;
};

}  // namespace ginfer::core::model