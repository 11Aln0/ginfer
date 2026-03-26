#include <glog/logging.h>
#include "ginfer/core/layer/layer.h"
#include "ginfer/core/layer/transformer/layer.h"
#include "ginfer/core/tensor/tensor.h"

namespace ginfer::core::layer::transformer {

AttentionLayer::AttentionLayer(
    DeviceType dev_type, std::string layer_name, int num_heads, int num_kv_heads, int head_dim)
    : Layer(dev_type, std::move(layer_name)),
      rope_op(dev_type),
      gqa_op(dev_type),
      gqa_varlen_op(dev_type, 16 /* TODO: hardcoded paged_block_size */),
      store_kv_op(dev_type),
      q_proj(dev_type, "q_proj"),
      k_proj(dev_type, "k_proj"),
      v_proj(dev_type, "v_proj"),
      o_proj(dev_type, "o_proj"),
      num_heads_(num_heads),
      num_kv_heads_(num_kv_heads),
      head_dim_(head_dim) {}

Result<void, std::string> AttentionLayer::forwardWithKVCache(const core::InferContext& ctx,
                                                             const TensorRef& q,
                                                             const TensorRef& k,
                                                             const TensorRef& v,
                                                             TensorRef output) {
  CHECK(ctx.slot_mapping.has_value())
      << "slot_mapping is required in InferContext for AttentionLayer.";
  CHECK(ctx.cu_seqlens_q.has_value() && ctx.cu_seqlens_kv.has_value())
      << "cu_seqlens_q, cu_seqlens_kv are required in InferContext for AttentionLayer.";
  CHECK(ctx.block_tables.has_value())
      << "block_tables is required in InferContext for AttentionLayer.";
  CHECK(k_cache_ != nullptr && v_cache_ != nullptr)
      << "KV cache tensors must be set before calling AttentionLayer forward.";
  CHECK(ctx.slot_mapping.value()->shape()[0] == k->shape()[0])
      << "slot_mapping length must match the sequence length of k/v tensors.";

  RETURN_ON_ERR(store_kv_op.run(
      ctx, {k.get(), v.get(), k_cache_.get(), v_cache_.get(), ctx.slot_mapping.value().get()}, {}));
  return gqa_varlen_op.run(ctx,
                           {q.get(), k_cache_.get(), v_cache_.get(), ctx.cu_seqlens_q.value().get(),
                            ctx.cu_seqlens_kv.value().get(), ctx.block_tables.value().get()},
                           {output.get()});
}

Result<void, std::string> AttentionLayer::forwardWithoutKVCache(const core::InferContext& ctx,
                                                                const TensorRef& q,
                                                                const TensorRef& k,
                                                                const TensorRef& v,
                                                                TensorRef output) {
  return gqa_op.run(ctx, {q.get(), k.get(), v.get()}, {output.get()});
}

Result<void, std::string> AttentionLayer::forward(const core::InferContext& ctx,
                                                  const std::vector<TensorRef>& inputs,
                                                  TensorRef output) {
  CHECK(inputs.size() == 4) << "AttentionLayer requires exactly 4 input tensors.";
  const auto& hidden_state = inputs[0];  // [seq_len, hidden_size]
  const auto& positions = inputs[1];     // [seq_len]
  const auto& sin_cache = inputs[2];     // [seq_len, head_dim / 2]
  const auto& cos_cache = inputs[3];     // [seq_len, head_dim / 2]

  int64_t seq_len = hidden_state->shape()[0];

  const TensorRef q = intermediates_.q_proj_out->slice(0, 0, seq_len)
                          ->reshape({
                              seq_len,
                              num_heads_,
                              head_dim_,
                          });
  const TensorRef k = intermediates_.k_proj_out->slice(0, 0, seq_len)
                          ->reshape({
                              seq_len,
                              num_kv_heads_,
                              head_dim_,
                          });
  const TensorRef v = intermediates_.v_proj_out->slice(0, 0, seq_len)
                          ->reshape({
                              seq_len,
                              num_kv_heads_,
                              head_dim_,
                          });
  TensorRef gqa_out = intermediates_.gqa_out->slice(0, 0, seq_len);

  RETURN_ON_ERR(q_proj.forward(ctx, {hidden_state}, q));
  RETURN_ON_ERR(k_proj.forward(ctx, {hidden_state}, k));
  RETURN_ON_ERR(v_proj.forward(ctx, {hidden_state}, v));

  RETURN_ON_ERR(
      rope_op.run(ctx, {q.get(), positions.get(), sin_cache.get(), cos_cache.get()}, {q.get()}));
  RETURN_ON_ERR(
      rope_op.run(ctx, {k.get(), positions.get(), sin_cache.get(), cos_cache.get()}, {k.get()}));

  if (ctx.block_tables.has_value()) {
    RETURN_ON_ERR(forwardWithKVCache(ctx, q, k, v, gqa_out));
  } else {
    RETURN_ON_ERR(forwardWithoutKVCache(ctx, q, k, v, gqa_out));
  }

  gqa_out = gqa_out->reshape(hidden_state->shape());
  return o_proj.forward(ctx, {gqa_out}, output);
}

// void AttentionLayer::reset() { kv_cache_.offset = 0; }

void AttentionLayer::setWeight(const Weight& weight) {
  auto w = weight;
  q_proj.setWeight(w.q_w);
  q_proj.setBias(w.q_b);
  k_proj.setWeight(w.k_w);
  k_proj.setBias(w.k_b);
  v_proj.setWeight(w.v_w);
  v_proj.setBias(w.v_b);
  o_proj.setWeight(w.o_w);
}

void AttentionLayer::setIntermediates(const Intermediates& intermediates) {
  intermediates_ = intermediates;
}

Result<void, std::string> AttentionLayer::toDevice(DeviceType dev_type) {
  RETURN_ON_ERR(rope_op.toDevice(dev_type));
  RETURN_ON_ERR(gqa_op.toDevice(dev_type));
  RETURN_ON_ERR(gqa_varlen_op.toDevice(dev_type));
  RETURN_ON_ERR(store_kv_op.toDevice(dev_type));
  RETURN_ON_ERR(q_proj.toDevice(dev_type));
  RETURN_ON_ERR(k_proj.toDevice(dev_type));
  RETURN_ON_ERR(v_proj.toDevice(dev_type));
  RETURN_ON_ERR(o_proj.toDevice(dev_type));
  return Layer::toDevice(dev_type);
}

void AttentionLayer::setKVCache(TensorRef& k_cache, TensorRef& v_cache) {
  k_cache_ = k_cache;
  v_cache_ = v_cache;
}

}  // namespace ginfer::core::layer::transformer