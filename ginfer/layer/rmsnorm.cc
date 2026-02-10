#include <glog/logging.h>
#include "ginfer/layer/layer.h"

namespace ginfer::layer {

RMSNormLayer::RMSNormLayer(DeviceType dev_type, std::string layer_name, float epsilon)
    : LayerWithParam(dev_type, std::move(layer_name)), norm_op_(dev_type, epsilon) {}

Status RMSNormLayer::forward(const std::vector<TensorRef>& inputs, TensorRef output) {
  CHECK_EQ(inputs.size(), 1) << "RMSNormLayer requires exactly 1 input tensor.";
  return norm_op_.run({inputs[0].get(), gamma_.get()}, {output.get()});
}

void RMSNormLayer::setWeight(const TensorRef& gamma) { gamma_ = gamma; }

std::vector<TensorRef> RMSNormLayer::getWeights() { return {gamma_}; }

}  // namespace ginfer::layer