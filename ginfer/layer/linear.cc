#include <glog/logging.h>
#include "ginfer/layer/layer.h"

namespace ginfer::layer {

LinearLayer::LinearLayer(DeviceType dev_type, std::string layer_name)
    : Layer(dev_type, std::move(layer_name)), mm_op_(dev_type) {}

void LinearLayer::setWeight(const TensorRef& weight) {
  const auto& shape = weight->shape();
  const auto& strides = weight->strides();
  CHECK_EQ(shape.ndim(), 2) << "Weight tensor must be 2D.";
  if (strides[1] == 1) {
    // row major
    weight_ = weight->permute({1, 0});  // [out_features, in_features] -> [in_features, out_features]
  } else {
    weight_ = weight;
  }
}

void LinearLayer::setBias(const TensorRef& bias) { bias_ = bias; }

Status LinearLayer::forward(const std::vector<TensorRef>& inputs, TensorRef output) {
  CHECK_EQ(inputs.size(), 1) << "LinearLayer requires exactly 1 input tensor.";

  std::vector<const Tensor*> mm_inputs = {inputs[0].get(), weight_.get()};
  if (bias_ != nullptr) {
    mm_inputs.push_back(bias_.get());
  }

  return mm_op_.run(mm_inputs, {output.get()});
}

Status LinearLayer::toDevice(DeviceType dev_type) {
  weight_->toDevice(dev_type);
  if (bias_ != nullptr) {
    bias_->toDevice(dev_type);
  }
  return mm_op_.toDevice(dev_type);
}

}  // namespace ginfer::layer