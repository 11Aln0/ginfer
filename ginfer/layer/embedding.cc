#include "ginfer/layer/layer.h"

namespace ginfer::layer {

EmbeddingLayer::EmbeddingLayer(DeviceType dev_type, std::string layer_name)
    : Layer(dev_type, std::move(layer_name)), embed_op_(dev_type) {}

void EmbeddingLayer::setWeight(const TensorRef& weight) { weight_ = weight; }

Result<void, std::string> EmbeddingLayer::forward(const std::vector<TensorRef>& inputs, TensorRef output) {
  CHECK_EQ(inputs.size(), 1) << "EmbeddingLayer requires exactly 1 input tensor.";
  return embed_op_.run({inputs[0].get(), weight_.get()}, {output.get()});
}

Result<void, std::string> EmbeddingLayer::toDevice(DeviceType dev_type) {
  weight_->toDevice(dev_type);
  return embed_op_.toDevice(dev_type);
}

}  // namespace ginfer::layer