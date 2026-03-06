#include "ginfer/model/model.h"
#include "ginfer/common/context.h"
#include "ginfer/memory/allocator_factory.h"

namespace ginfer::model {

LlamaArchModel::LlamaArchModel(LlamaArchModelConfig config, common::DeviceType dev_type)
    : config_(config),
      dev_type_(dev_type),
      embed_tokens(dev_type, "embed_tokens"),
      final_rmsnorm(dev_type, "final_rmsnorm", config.rms_norm_eps),
      lm_head(dev_type, "lm_head"),
      argmax_op(dev_type),
      Model(config, dev_type) {
  // initialize encoder layers
  for (size_t i = 0; i < config.nlayer; ++i) {
    encoder_layers.emplace_back(dev_type, "encoder_layer_" + std::to_string(i), config.rms_norm_eps, config.num_heads,
                                config.num_kv_heads, config.head_dim);
  }
}

Result<void, std::string> LlamaArchModel::lazyAllocKVCache() {
  if (kv_cache_allocated_) return Ok<void>();
  kv_cache_.offset = 0;
  const auto& c = config_;
  int64_t max_seq_len = static_cast<int64_t>(c.max_seq_len);
  // init k,v cache
  for (int i = 0; i < config_.nlayer; ++i) {
    TensorRef k, v;
    ASSIGN_OR_RETURN(k,
                     Tensor::create(config_.dtype, tensor::Shape{max_seq_len, c.num_kv_heads, c.head_dim}, dev_type_));
    ASSIGN_OR_RETURN(v,
                     Tensor::create(config_.dtype, tensor::Shape{max_seq_len, c.num_kv_heads, c.head_dim}, dev_type_));
    kv_cache_.k.push_back(std::move(k));
    kv_cache_.v.push_back(std::move(v));
  }
  kv_cache_allocated_ = true;
  return Ok<void>();
}

Result<void, std::string> LlamaArchModel::lazyAllocIntermediates() {
  if (intermediates_allocated_) return Ok<void>();

  using memory::DeviceType;
  using tensor::DataType;
  using tensor::Shape;
  using tensor::Tensor;

  int64_t max_seq_len = static_cast<int64_t>(config_.max_seq_len);
  int64_t head_dim = static_cast<int64_t>(config_.head_dim);
  auto dtype = config_.dtype;
  auto dev_type = dev_type_;

  auto& i = intermediates_;
  DECLARE_OR_RETURN(t0, Tensor::create(dtype, Shape{max_seq_len, config_.hidden_size}, dev_type));
  ASSIGN_OR_RETURN(i.sin, Tensor::create(DataType::kDataTypeFloat32, Shape{max_seq_len, head_dim / 2}, dev_type));
  ASSIGN_OR_RETURN(i.cos, Tensor::create(DataType::kDataTypeFloat32, Shape{max_seq_len, head_dim / 2}, dev_type));
  ASSIGN_OR_RETURN(i.embed_out, Tensor::create(dtype, Shape{max_seq_len, config_.hidden_size}, dev_type));
  ASSIGN_OR_RETURN(i.lm_head_out, Tensor::create(dtype, Shape{1, config_.vocab_size}, dev_type));
  ASSIGN_OR_RETURN(i.argmax_out, Tensor::create(DataType::kDataTypeInt64, Shape{1}, dev_type));
  i.norm_out = t0;

  DECLARE_OR_RETURN(t1, Tensor::create(dtype, Shape{max_seq_len, config_.num_heads * head_dim}, dev_type));
  layer::transformer::AttentionLayer::Intermediates attn_intermediates = {
      .q_proj_out = t1,
      .gqa_out = t1,
  };

  DECLARE_OR_RETURN(t2, Tensor::create(dtype, Shape{max_seq_len, config_.intermediate_size}, dev_type));
  DECLARE_OR_RETURN(up_out, Tensor::create(dtype, Shape{max_seq_len, config_.intermediate_size}, dev_type));
  layer::transformer::FeedForwardLayer::Intermediates mlp_intermediates = {
      .gate_out = t2,
      .up_out = up_out,
      .swiglu_out = t2,
  };

  DECLARE_OR_RETURN(attn_out, Tensor::create(dtype, Shape{max_seq_len, config_.hidden_size}, dev_type));
  layer::transformer::EncoderLayer::Intermediates encoder_intermediates = {
      .attn = attn_intermediates,
      .mlp = mlp_intermediates,
      .norm_out = t0,
      .attn_out = attn_out,
  };

  for (auto& e : encoder_layers) {
    e.setIntermediates(encoder_intermediates);
  }

  intermediates_allocated_ = true;
  return Ok<void>();
}

Result<std::pair<TensorRef, TensorRef>, std::string> LlamaArchModel::getPositionEmbedding(
    TensorRef sin_cache, TensorRef cos_cache, std::pair<int64_t, int64_t> pos_id_range) {
  auto [start, end] = pos_id_range;
  auto sin = sin_cache->slice(0, 0, end - start + 1);
  auto cos = cos_cache->slice(0, 0, end - start + 1);
  auto allocator = memory::getDeviceAllocator<memory::PooledAllocStrategy>(memory::DeviceType::kDeviceCPU);
  DECLARE_OR_RETURN(pos_ids, Tensor::create(tensor::DataType::kDataTypeInt64, tensor::Shape{2}, allocator));
  pos_ids->data<int64_t>()[0] = start;
  pos_ids->data<int64_t>()[1] = end;
  getRotaryEmbeddingOp().run(common::InferContext{}, {pos_ids.get()}, {sin.get(), cos.get()});
  return Ok(std::pair<TensorRef, TensorRef>{sin, cos});
}

Result<void, std::string> LlamaArchModel::forward(const TensorRef input_ids, std::pair<int64_t, int64_t> pos_id_range,
                                                  TensorRef output) {
  int64_t seq_len = input_ids->shape()[0];
  DECLARE_OR_RETURN(sin_cos, getPositionEmbedding(intermediates_.sin, intermediates_.cos, pos_id_range));
  auto [sin, cos] = sin_cos;

  auto embed_out = intermediates_.embed_out->slice(0, 0, seq_len);
  common::InferContext infer_ctx;
  RETURN_ON_ERR(embed_tokens.forward(infer_ctx, {input_ids}, embed_out));
  TensorRef hidden_state = embed_out;
  for (int i = 0; i < config_.nlayer; ++i) {
    auto& encoder = encoder_layers[i];
    auto k_cache = kv_cache_.k[i]->slice(0, 0, kv_cache_.offset + seq_len);
    auto v_cache = kv_cache_.v[i]->slice(0, 0, kv_cache_.offset + seq_len);
    RETURN_ON_ERR(encoder.forward(infer_ctx, {hidden_state, sin, cos, k_cache, v_cache}, hidden_state));
  }
  kv_cache_.offset += seq_len;
  return final_rmsnorm.forward(infer_ctx, {hidden_state}, output);
}

Result<void, std::string> LlamaArchModel::toDevice(common::DeviceType dev_type) {
  dev_type_ = dev_type;
  RETURN_ON_ERR(embed_tokens.toDevice(dev_type));
  RETURN_ON_ERR(getRotaryEmbeddingOp().toDevice(dev_type));
  for (auto& encoder : encoder_layers) {
    RETURN_ON_ERR(encoder.toDevice(dev_type));
  }
  RETURN_ON_ERR(final_rmsnorm.toDevice(dev_type));
  RETURN_ON_ERR(lm_head.toDevice(dev_type));
  return argmax_op.toDevice(dev_type);
}

Result<int32_t, std::string> LlamaArchModel::predict(const tensor::TensorRef token_ids,
                                                     std::pair<int64_t, int64_t> pos_id_range) {
  RETURN_ON_ERR(lazyAllocIntermediates());
  RETURN_ON_ERR(lazyAllocKVCache());

  int64_t seq_len = token_ids->shape()[0];
  CHECK_LE(seq_len, static_cast<int64_t>(config_.max_seq_len)) << "Input token_ids length exceeds max_seq_len.";
  auto input_ids = token_ids->slice(0, 0, seq_len);
  input_ids->toDevice(memory::getDeviceAllocator<memory::PooledAllocStrategy>(dev_type_));

  // forward
  auto norm_out = intermediates_.norm_out->slice(0, 0, seq_len);
  auto lm_head_out = intermediates_.lm_head_out;
  RETURN_ON_ERR(forward(input_ids, pos_id_range, norm_out));
  norm_out = norm_out->slice(0, seq_len - 1, seq_len);  // only need the last token's hidden state for LM head
  RETURN_ON_ERR(lm_head.forward(common::InferContext{}, {norm_out}, lm_head_out));

  // get next token id (argmax)
  auto argmax_out = intermediates_.argmax_out->slice(0, 0, 1);
  RETURN_ON_ERR(argmax_op.run(common::InferContext{}, {lm_head_out.get()}, {argmax_out.get()}));
  argmax_out->toDevice(memory::getDeviceAllocator<memory::PooledAllocStrategy>(memory::DeviceType::kDeviceCPU));

  return Ok(argmax_out->data<int32_t>()[0]);
}

}  // namespace ginfer::model
