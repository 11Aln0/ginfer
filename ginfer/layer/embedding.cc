#include "ginfer/layer/layer.h"

namespace ginfer::layer {

EmbeddingLayer::EmbeddingLayer(DeviceType dev_type, std::string layer_name)
    : LayerWithParam(dev_type, std::move(layer_name)), embed_op_(dev_type) {}

void EmbeddingLayer::setWeight(const TensorRef& weight) { weight_ = weight; }

Status EmbeddingLayer::forward(const std::vector<TensorRef>& inputs, TensorRef output) {
  CHECK_EQ(inputs.size(), 1) << "EmbeddingLayer requires exactly 1 input tensor.";
  return embed_op_.run({inputs[0].get(), weight_.get()}, {output.get()});
}

std::vector<TensorRef> EmbeddingLayer::getWeights() { return {weight_}; }

}  // namespace ginfer::layer