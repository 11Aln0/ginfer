#pragma once

#include <memory>
#include <vector>
#include "ginfer/common/errors.h"
#include "ginfer/layer/layer.h"
#include "ginfer/layer/transformer/layer.h"
#include "ginfer/memory/allocator_factory.h"
#include "ginfer/model/loader/safetensor_loader.h"
#include "ginfer/op/op.h"
#include "ginfer/tensor/tensor.h"

namespace ginfer::model {

using ginfer::tensor::Tensor;
using ginfer::tensor::TensorRef;

struct ModelConfig {
 public:
  tensor::DataType dtype;

  int vocab_size;
  size_t max_seq_len;

  std::vector<int32_t> eos_token_ids;
};

class Model {
 public:
  Model() = delete;

  explicit Model(ModelConfig config, common::DeviceType dev_type) : config_(std::move(config)), dev_type_(dev_type) {}

  virtual Result<int32_t, std::string> predict(const tensor::TensorRef token_ids,
                                               std::pair<int64_t, int64_t> pos_id_range) = 0;

  virtual Result<void, std::string> toDevice(common::DeviceType dev_type) {
    dev_type_ = dev_type;
    return Ok<void>();
  }

  int getVocabSize() const { return config_.vocab_size; }

  bool isEosToken(int32_t token_id) const {
    for (auto id : config_.eos_token_ids) {
      if (id == token_id) return true;
    }
    return false;
  }

 private:
  ModelConfig config_;
  common::DeviceType dev_type_;
};

struct LlamaArchModelConfig : public ModelConfig {
  int nlayer;
  int hidden_size;
  int num_heads;
  int num_kv_heads;
  int head_dim;
  int intermediate_size;

  float rms_norm_eps;
  float rope_theta;

  bool tie_word_embeddings;
};

class LlamaArchModel : public Model {
 public:
  LlamaArchModel(LlamaArchModelConfig config, common::DeviceType dev_type);

  Result<int32_t, std::string> predict(const tensor::TensorRef token_ids,
                                       std::pair<int64_t, int64_t> pos_id_range) override;

  Result<void, std::string> toDevice(common::DeviceType dev_type) override;

 private:
  struct Intermediates {
    TensorRef sin;          // [max_seq_len, head_dim]
    TensorRef cos;          // [max_seq_len, head_dim]
    TensorRef embed_out;    // [max_seq_len, hidden_size]
    TensorRef norm_out;     // [max_seq_len, hidden_size]
    TensorRef lm_head_out;  // [max_seq_len, vocab_size]
    TensorRef argmax_out;   // [1]
  };

  struct KVCache {
    std::vector<TensorRef> k;  // [max_seq_len, num_kv_heads * head_dim] * nlayer
    std::vector<TensorRef> v;  // [max_seq_len, num_kv_heads * head_dim] * nlayer
    int64_t offset;
  };

 private:
  // mem
  Result<void, std::string> lazyAllocIntermediates();
  Result<void, std::string> lazyAllocKVCache();

  Result<std::pair<TensorRef, TensorRef>, std::string> getPositionEmbedding(TensorRef sin_cache, TensorRef cos_cache,
                                                                            std::pair<int64_t, int64_t> pos_id_range);

  // forward
  Result<void, std::string> forward(const TensorRef input_ids, std::pair<int64_t, int64_t> pos_id_range,
                                    TensorRef output);

 private:
  LlamaArchModelConfig config_;
  common::DeviceType dev_type_;
  bool intermediates_allocated_ = false;
  bool kv_cache_allocated_ = false;

  Intermediates intermediates_;
  KVCache kv_cache_;

 protected:
  virtual op::Op& getRotaryEmbeddingOp() = 0;

 protected:
  op::ArgmaxOp argmax_op;
  layer::EmbeddingLayer embed_tokens;
  std::vector<layer::transformer::EncoderLayer> encoder_layers;
  layer::RMSNormLayer final_rmsnorm;
  layer::LinearLayer lm_head;
};

}  // namespace ginfer::model