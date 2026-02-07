#pragma once

#include <memory>
#include "ginfer/common/errors.h"
#include "ginfer/op/op.h"
#include "ginfer/tensor/tensor.h"

namespace ginfer::model {

class AttentionLayer : ginfer::op::Layer {};

class FeedForwardLayer {};

class EncoderLayer {};

class Qwen2Model {
 private:
  using error::Status;
  using TensorRef = std::shared_ptr<tensor::Tensor>;

  struct Qwen2Config {
    tensor::DataType dtype;

    size_t nlayer;
    size_t hidden_size;
    size_t num_heads;
    size_t num_kv_heads;
    size_t head_dim;
    size_t intermediate_size;
    size_t vocab_size;
    size_t max_seq_len;

    float rms_norm_eps;
    float rope_theta;

    int64_t eos_token_id;
  };

  struct Qwen2Weights {
    // embedding
    TensorRef in_embed_w;

    // encoder layer
    std::vector<TensorRef> attn_norm_w;  // [hidden_size]
    std::vector<TensorRef> attn_q_w;     // [hidden_size, num_heads * head_dim]
    std::vector<TensorRef> attn_q_b;
    std::vector<TensorRef> attn_k_w;  // [hidden_size, num_heads * head_dim]
    std::vector<TensorRef> attn_k_b;
    std::vector<TensorRef> attn_v_w;  // [hidden_size, num_kv_heads * head_dim]
    std::vector<TensorRef> attn_v_b;
    std::vector<TensorRef> attn_o_w;    // [hidden_size, num_heads * head_dim]
    std::vector<TensorRef> mlp_norm_w;  // [hidden_size, intermediate_size]
    std::vector<TensorRef> mlp_gate_w;  // [hidden_size, intermediate_size]
    std::vector<TensorRef> mlp_up_w;
    std::vector<TensorRef> mlp_down_w;

    TensorRef out_norm_w;
    // lm head
    TensorRef lm_head_w;  // [hidden_size, vocab_size]
  };

  struct Qwen2InternalBuffers {
    TensorRef token_ids;  // [max_seq_len]

    TensorRef embed_out;     // [max_seq_len, hidden_size]
    TensorRef rms_norm_out;  // for decoder rmsx2 + lm_head rms [max_seq_len, hidden_size]

    // attention
    TensorRef q_out;       // [max_seq_len, num_heads * head_dim]
    TensorRef attn_out;    // [max_seq_len, num_heads * head_dim]
    TensorRef o_proj_out;  // [max_seq_len, hidden_size]
    // no need for redisual

    // feed-forward
    TensorRef mlp_gate_out;  // [max_seq_len, inter_size] (reused for swiglu out)
    TensorRef mlp_up_out;    // [max_seq_len, inter_size]
    TensorRef mlp_down_out;  // [max_seq_len, hidden_size]

    TensorRef lm_head_out;  // [max_seq_len, vocab_size]
  };

  struct Qwen2KVCache {
    std::vector<TensorRef> k;  // [max_seq_len, num_kv_heads * head_dim] * nlayer
    std::vector<TensorRef> v;  // [max_seq_len, num_kv_heads * head_dim] * nlayer
  };

 public:
  Status predict(const tensor::Tensor& token_ids, std::pair<int64_t, int64_t> pos_id_range, int64_t& next_token_id);

 private:
  // mem
  void initEncoderLayerWeight(size_t layer);
  void initEncoderLayersWeight();
  void initWeights();
  void initInternalBuffers();
  void initKVCache();

  // forward
  TensorRef forward(const TensorRef input_ids);
  TensorRef forwardEmbedding(const TensorRef input_ids);
  TensorRef forwardRMSNorm(const TensorRef input, const TensorRef weight);
  TensorRef forwardAttention(const TensorRef input, int layer);
  TensorRef forwardFeedForward(const TensorRef input, int layer);
  TensorRef forwardEncoderLayer(const TensorRef input, int layer);
  TensorRef forwardLMHead(const TensorRef input);

 private:
  Qwen2Config config_;
  Qwen2Weights weights_;
  Qwen2InternalBuffers internal_buffers_;
  Qwen2KVCache kv_cache_;

 private:
};

}  // namespace ginfer::model