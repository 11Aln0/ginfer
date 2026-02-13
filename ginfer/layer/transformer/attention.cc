#include <glog/logging.h>
#include "ginfer/layer/layer.h"
#include "ginfer/layer/transformer/layer.h"
#include "ginfer/tensor/tensor.h"

namespace ginfer::layer::transformer {

AttentionLayer::AttentionLayer(DeviceType dev_type, std::string layer_name, int num_heads, int num_kv_heads,
                               int head_dim)
    : Layer(dev_type, std::move(layer_name)),
      rope_op(dev_type),
      gqa_op(dev_type),
      q_proj(dev_type, "q_proj"),
      k_proj(dev_type, "k_proj"),
      v_proj(dev_type, "v_proj"),
      o_proj(dev_type, "o_proj"),
      num_heads_(num_heads),
      num_kv_heads_(num_kv_heads),
      head_dim_(head_dim) {}

Status AttentionLayer::forward(const std::vector<TensorRef>& inputs, TensorRef output) {
  CHECK(inputs.size() == 5) << "AttentionLayer requires exactly 5 input tensors.";
  const auto& hidden_state = inputs[0];  // [seq_len, hidden_size]
  const auto& sin_cache = inputs[1];     // [seq_len, head_dim]
  const auto& cos_cache = inputs[2];     // [seq_len, head_dim]
  const auto& k_cache = inputs[3];       // [kv_cache_off + seq_len, num_kv_heads, head_dim]
  const auto& v_cache = inputs[4];       // [kv_cache_off + seq_len, num_kv_heads, head_dim]

  int64_t q_seq_len = hidden_state->shape()[0];
  int64_t kv_seq_len = k_cache->shape()[0];

  const TensorRef q = intermediates_.q_proj_out->slice(0, 0, q_seq_len)
                          ->reshape(tensor::Shape({
                              q_seq_len,
                              num_heads_,
                              head_dim_,
                          }));
  const TensorRef k = k_cache->slice(0, kv_seq_len - q_seq_len, kv_seq_len);
  const TensorRef v = v_cache->slice(0, kv_seq_len - q_seq_len, kv_seq_len);
  TensorRef gqa_out = intermediates_.gqa_out->slice(0, 0, q_seq_len);

  RETURN_ON_ERROR(q_proj.forward({hidden_state}, q));
  RETURN_ON_ERROR(k_proj.forward({hidden_state}, k));
  RETURN_ON_ERROR(v_proj.forward({hidden_state}, v));

  RETURN_ON_ERROR(rope_op.run({q.get(), sin_cache.get(), cos_cache.get()}, {q.get()}));
  RETURN_ON_ERROR(rope_op.run({k.get(), sin_cache.get(), cos_cache.get()}, {k.get()}));
  RETURN_ON_ERROR(gqa_op.run({q.get(), k_cache.get(), v_cache.get()}, {gqa_out.get()}));

  gqa_out = gqa_out->reshape(hidden_state->shape());  // reshape back to [seq_len, num_heads * head_dim]
  return o_proj.forward({gqa_out}, output);
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

void AttentionLayer::setIntermediates(const Intermediates& intermediates) { intermediates_ = intermediates; }

Status AttentionLayer::toDevice(DeviceType dev_type) {
  RETURN_ON_ERROR(rope_op.toDevice(dev_type));
  RETURN_ON_ERROR(gqa_op.toDevice(dev_type));
  RETURN_ON_ERROR(q_proj.toDevice(dev_type));
  RETURN_ON_ERROR(k_proj.toDevice(dev_type));
  RETURN_ON_ERROR(v_proj.toDevice(dev_type));
  return o_proj.toDevice(dev_type);
}

}  // namespace ginfer::layer::transformer