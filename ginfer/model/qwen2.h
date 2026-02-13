#pragma once

#include <memory>
#include "ginfer/common/errors.h"
#include "ginfer/layer/layer.h"
#include "ginfer/layer/transformer/layer.h"
#include "ginfer/memory/allocator_factory.h"
#include "ginfer/model/safetensor_loader.h"
#include "ginfer/op/op.h"
#include "ginfer/tensor/tensor.h"

namespace ginfer::model {

using error::Status;
using ginfer::tensor::Tensor;
using ginfer::tensor::TensorRef;

struct Qwen2Config {
  tensor::DataType dtype;

  int nlayer;
  int hidden_size;
  int num_heads;
  int num_kv_heads;
  int head_dim;
  int intermediate_size;
  int vocab_size;
  size_t max_seq_len;

  float rms_norm_eps;
  float rope_theta;

  int64_t eos_token_id;
};

class Qwen2Model;

class Qwen2ModelLoader {
  using AttentionWeight = layer::transformer::AttentionLayer::Weight;
  using EncoderWeight = layer::transformer::EncoderLayer::Weight;
  using FeedForwardWeight = layer::transformer::FeedForwardLayer::Weight;

 public:
  explicit Qwen2ModelLoader(std::string model_path);

  std::shared_ptr<Qwen2Model> load();

 private:
  Qwen2Config loadConfig();
  AttentionWeight loadAttentionWeight(const std::string& prefix);
  FeedForwardWeight loadFeedForwardWeight(const std::string& prefix);
  EncoderWeight loadEncoderLayerWeight(int layer_idx);

 private:
  std::string model_path_;
  SafeTensorLoader weight_loader;
};

class Qwen2Model {
  struct Intermediates {
    TensorRef sin;          // [max_seq_len, head_dim]
    TensorRef cos;          // [max_seq_len, head_dim]
    TensorRef embed_out;    // [max_seq_len, hidden_size]
    TensorRef norm_out;     // [max_seq_len, hidden_size]
    TensorRef lm_head_out;  // [max_seq_len, vocab_size]
    TensorRef argmax_out;   // [1]
  };

  struct Qwen2KVCache {
    std::vector<TensorRef> k;  // [max_seq_len, num_kv_heads * head_dim] * nlayer
    std::vector<TensorRef> v;  // [max_seq_len, num_kv_heads * head_dim] * nlayer
    int64_t offset;
  };

 public:
  Qwen2Model(Qwen2Config config, common::DeviceType dev_type = common::DeviceType::kDeviceCPU);

  Status predict(const tensor::Tensor& token_ids, std::pair<int64_t, int64_t> pos_id_range, int64_t& next_token_id);

  void toDevice(common::DeviceType dev_type);

  friend class Qwen2ModelLoader;

 private:
  struct KVCache {
    TensorRef k;
    TensorRef v;
    int64_t offset;
  };

  // TODO select by device
  using InputTensorAllocator = ginfer::memory::GlobalCUDAAllocator<ginfer::memory::cuda::PooledAllocStrategy>;

 private:
  // mem
  void lazyAllocIntermediates();
  void initKVCache();
  std::pair<TensorRef, TensorRef> getPositionEmbedding(std::pair<int64_t, int64_t> pos_id_range);

  // forward
  Status forward(const TensorRef input_ids, std::pair<int64_t, int64_t> pos_id_range, TensorRef output);

 private:
  Qwen2Config config_;
  common::DeviceType dev_type_;

  Intermediates intermediates_;
  Qwen2KVCache kv_cache_;
  bool intermediates_allocated_ = false;

  op::RotaryEmbeddingOp rotary_emb;
  op::ArgmaxOp argmax_op;
  layer::EmbeddingLayer embed_tokens;
  std::vector<layer::transformer::EncoderLayer> encoder_layers;
  layer::RMSNormLayer final_rmsnorm;
  layer::LinearLayer lm_head;
};

}  // namespace ginfer::model