#include "layer.h"
#include <glog/logging.h>
#include "ginfer/common/errors.h"

namespace ginfer::op {

BaseLayer::BaseLayer(DeviceType dev_type, LayerType layer_type, std::string layer_name)
    : dev_type_(dev_type), layer_type_(layer_type), layer_name_(std::move(layer_name)) {}

LayerType BaseLayer::layerType() const { return layer_type_; }

DeviceType BaseLayer::getDeviceType() const { return dev_type_; }

Status BaseLayer::toDevice(DeviceType dev_type) {
  dev_type_ = dev_type;
  return ginfer::error::Success();
}

Status LayerWithParam::toDevice(DeviceType dev_type) {
  for (auto weight : weights_) {
    weight->toDevice(dev_type);
  }
  return BaseLayer::toDevice(dev_type);
}

void LayerWithParam::resetWeightSize(size_t size) { weights_.resize(size); }

void LayerWithParam::setWeight(int32_t idx, std::shared_ptr<Tensor> weight) { weights_[idx] = weight; }

std::shared_ptr<Tensor> LayerWithParam::getWeight(int32_t idx) {
  CHECK_GT(weights_.size(), idx);
  CHECK_GT(idx, -1);
  return weights_[idx];
}

std::vector<std::shared_ptr<Tensor>> LayerWithParam::getWeights() { return weights_; }

};  // namespace ginfer::op
