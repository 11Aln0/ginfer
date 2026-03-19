#include <glog/logging.h>
#include "ginfer/core/layer/transformer/layer.h"

namespace ginfer::core::layer::transformer {

EncoderLayer::EncoderLayer(DeviceType dev_type,
                           std::string layer_name,
                           float rms_norm_eps,
                           int num_heads,
                           int num_kv_heads,
                           int head_dim)
    : Layer(dev_type, std::move(layer_name)),
      self_attn(dev_type, "self_attn", num_heads, num_kv_heads, head_dim),
      mlp(dev_type, "mlp"),
      mlp_norm(dev_type, "mlp_norm", rms_norm_eps),
      attn_norm(dev_type, "attn_norm", rms_norm_eps),
      add(dev_type) {}

Result<void, std::string> EncoderLayer::forward(const core::InferContext& ctx,
                                                const std::vector<TensorRef>& inputs,
                                                TensorRef output) {
  CHECK(inputs.size() == 4) << "EncoderLayer requires exactly 4 input tensors.";
  const auto& hidden_state = inputs[0];  // [seq_len, hidden_size]
  const auto& positions = inputs[1];     // [seq_len]
  const auto& sin_cache = inputs[2];     // [seq_len, head_dim / 2]
  const auto& cos_cache = inputs[3];     // [seq_len, head_dim / 2]

  int64_t seq_len = hidden_state->shape()[0];

  TensorRef attn_out = intermediates_.attn_out->slice(0, 0, seq_len);
  TensorRef norm_out = intermediates_.norm_out->slice(0, 0, seq_len);

  RETURN_ON_ERR(attn_norm.forward(ctx, {hidden_state}, norm_out));
  RETURN_ON_ERR(self_attn.forward(ctx, {norm_out, positions, sin_cache, cos_cache}, attn_out));
  RETURN_ON_ERR(add.run(ctx, {hidden_state.get(), attn_out.get()}, {attn_out.get()}));
  RETURN_ON_ERR(mlp_norm.forward(ctx, {attn_out}, norm_out));
  RETURN_ON_ERR(mlp.forward(ctx, {norm_out}, output));

  return add.run(ctx, {attn_out.get(), output.get()}, {output.get()});
}

void EncoderLayer::setWeight(const Weight& weight) {
  auto w = weight;
  self_attn.setWeight(w.attn);
  mlp.setWeight(w.mlp);
  attn_norm.setWeight(w.attn_norm);
  mlp_norm.setWeight(w.mlp_norm);
}

void EncoderLayer::setIntermediates(const Intermediates& intermediates) {
  auto i = intermediates;
  self_attn.setIntermediates(i.attn);
  mlp.setIntermediates(i.mlp);
  intermediates_ = i;
}

Result<void, std::string> EncoderLayer::toDevice(DeviceType dev_type) {
  RETURN_ON_ERR(self_attn.toDevice(dev_type));
  RETURN_ON_ERR(mlp.toDevice(dev_type));
  RETURN_ON_ERR(mlp_norm.toDevice(dev_type));
  RETURN_ON_ERR(attn_norm.toDevice(dev_type));
  return add.toDevice(dev_type);
}

AttentionLayer& EncoderLayer::getAttentionLayer() { return self_attn; }

FeedForwardLayer& EncoderLayer::getFeedForwardLayer() { return mlp; }

RMSNormLayer& EncoderLayer::getAttnNormLayer() { return attn_norm; }

RMSNormLayer& EncoderLayer::getMLPNormLayer() { return mlp_norm; }

}  // namespace ginfer::core::layer::transformer