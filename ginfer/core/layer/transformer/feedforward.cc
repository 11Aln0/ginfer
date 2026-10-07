#include "ginfer/core/layer/transformer/layer.h"

namespace ginfer::core::layer::transformer {

FeedForwardLayer::FeedForwardLayer(DeviceType dev_type, std::string layer_name)
    : Layer(dev_type, std::move(layer_name)),
      gate_proj(dev_type, name() + ".gate_proj"),
      up_proj(dev_type, name() + ".up_proj"),
      swiglu_op(dev_type),
      down_proj(dev_type, name() + ".down_proj") {}

Result<void, std::string> FeedForwardLayer::forward(const core::InferContext& ctx,
                                                    const std::vector<TensorRef>& inputs,
                                                    TensorRef output) {
  const auto& input = inputs[0];  // [seq_len, hidden_size]
  int64_t seq_len = input->shape()[0];

  TensorRef gate_out = intermediates_.gate_out->slice(0, 0, seq_len);
  TensorRef up_out = intermediates_.up_out->slice(0, 0, seq_len);
  TensorRef swiglu_out = intermediates_.swiglu_out->slice(0, 0, seq_len);

  RETURN_ON_ERR(gate_proj.forward(ctx, {input}, gate_out));
  RETURN_ON_ERR(up_proj.forward(ctx, {input}, up_out));
  RETURN_ON_ERR(swiglu_op.run(ctx, {gate_out.get(), up_out.get()}, {swiglu_out.get()}));
  return down_proj.forward(ctx, {swiglu_out}, output);
}

void FeedForwardLayer::setIntermediates(const Intermediates& intermediates) {
  intermediates_ = intermediates;
}

LinearLayer& FeedForwardLayer::gateProj() { return gate_proj; }
LinearLayer& FeedForwardLayer::upProj() { return up_proj; }
LinearLayer& FeedForwardLayer::downProj() { return down_proj; }

Result<void, std::string> FeedForwardLayer::toDevice(DeviceType dev_type) {
  RETURN_ON_ERR(gate_proj.toDevice(dev_type));
  RETURN_ON_ERR(up_proj.toDevice(dev_type));
  RETURN_ON_ERR(swiglu_op.toDevice(dev_type));
  RETURN_ON_ERR(down_proj.toDevice(dev_type));
  return Layer::toDevice(dev_type);
}

}  // namespace ginfer::core::layer::transformer