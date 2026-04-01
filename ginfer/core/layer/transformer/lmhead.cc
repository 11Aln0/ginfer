#include <glog/logging.h>
#include "ginfer/core/layer/transformer/layer.h"

namespace ginfer::core::layer::transformer {

LMHeadLayer::LMHeadLayer(DeviceType dev_type, std::string layer_name)
    : Layer(dev_type, std::move(layer_name)),
      lm_head_proj(dev_type, "lm_head_proj"),
      token_select_op(dev_type) {}

Result<void, std::string> LMHeadLayer::forwardWithKVCache(const core::InferContext& ctx,
                                                          const TensorRef& hidden_state,
                                                          TensorRef output) {
  TensorRef last_hidden_state = hidden_state;
  if (ctx.is_prefill) {
    CHECK(ctx.cu_seqlens_q.has_value())
        << "cu_seqlens_q is required in InferContext for LMHeadLayer.";
    CHECK(intermediates_.last_hidden_state != nullptr)
        << "LMHeadLayer intermediates must be set before calling forward.";
    auto batch_size = ctx.cu_seqlens_q.value()->shape()[0] - 1;
    last_hidden_state = intermediates_.last_hidden_state->slice(0, 0, batch_size);

    RETURN_ON_ERR(token_select_op.run(ctx, {hidden_state.get(), ctx.cu_seqlens_q.value().get()},
                                      {last_hidden_state.get()}));
  }

  return lm_head_proj.forward(ctx, {last_hidden_state}, output);
}

Result<void, std::string> LMHeadLayer::forwardWithoutKVCache(const core::InferContext& ctx,
                                                             const TensorRef& hidden_state,
                                                             TensorRef output) {
  int64_t seq_len = hidden_state->shape()[0];
  TensorRef last_hidden_state = hidden_state->slice(0, seq_len - 1, seq_len);
  return lm_head_proj.forward(ctx, {last_hidden_state}, output);
}

Result<void, std::string> LMHeadLayer::forward(const core::InferContext& ctx,
                                               const std::vector<TensorRef>& inputs,
                                               TensorRef output) {
  CHECK_EQ(inputs.size(), 1) << "LMHeadLayer requires exactly 1 input tensor.";
  if (ctx.block_tables.has_value()) {
    return forwardWithKVCache(ctx, inputs[0], output);
  }
  return forwardWithoutKVCache(ctx, inputs[0], output);
}

Result<void, std::string> LMHeadLayer::toDevice(DeviceType dev_type) {
  RETURN_ON_ERR(lm_head_proj.toDevice(dev_type));
  RETURN_ON_ERR(token_select_op.toDevice(dev_type));
  return Layer::toDevice(dev_type);
}

void LMHeadLayer::setWeight(const TensorRef& weight) { lm_head_proj.setWeight(weight); }

void LMHeadLayer::setIntermediates(const Intermediates& intermediates) {
  intermediates_ = intermediates;
}

}  // namespace ginfer::core::layer::transformer
