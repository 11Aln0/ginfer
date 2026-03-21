#include <glog/logging.h>
#include "ginfer/core/layer/layer.h"

namespace ginfer::core::layer {

LinearLayer::LinearLayer(DeviceType dev_type, std::string layer_name)
    : Layer(dev_type, std::move(layer_name)), mm_op_(dev_type) {}

void LinearLayer::setWeight(const TensorRef& weight) {
  const auto& shape = weight->shape();
  const auto& strides = weight->strides();
  CHECK_EQ(shape.ndim(), 2) << "Weight tensor must be 2D.";
  if (strides[1] == 1) {
    // row major
    weight_ =
        weight->permute({1, 0});  // [out_features, in_features] -> [in_features, out_features]
  } else {
    weight_ = weight;
  }
}

void LinearLayer::setBias(const TensorRef& bias) { bias_ = bias; }

Result<void, std::string> LinearLayer::forward(const core::InferContext& ctx,
                                               const std::vector<TensorRef>& inputs,
                                               TensorRef output) {
  CHECK_EQ(inputs.size(), 1) << "LinearLayer requires exactly 1 input tensor.";

  std::vector<const Tensor*> mm_inputs = {inputs[0].get(), weight_.get()};
  if (bias_ != nullptr) {
    mm_inputs.push_back(bias_.get());
  }

  return mm_op_.run(ctx, mm_inputs, {output.get()});
}

Result<void, std::string> LinearLayer::toDevice(DeviceType dev_type) {
  ASSIGN_OR_RETURN(weight_, weight_->toDevice(dev_type));
  if (bias_ != nullptr) {
    ASSIGN_OR_RETURN(bias_, bias_->toDevice(dev_type));
  }
  RETURN_ON_ERR(mm_op_.toDevice(dev_type));
  return Layer::toDevice(dev_type);
}

}  // namespace ginfer::core::layer