#include "layer.h"
#include <glog/logging.h>
#include "ginfer/common/errors.h"

namespace ginfer::layer {

BaseLayer::BaseLayer(DeviceType dev_type, std::string layer_name)
    : dev_type_(dev_type), layer_name_(std::move(layer_name)) {}

DeviceType BaseLayer::getDeviceType() const { return dev_type_; }

Status BaseLayer::toDevice(DeviceType dev_type) {
  dev_type_ = dev_type;
  return ginfer::error::Success();
}

// Status LayerWithParam::toDevice(DeviceType dev_type) {
//   for (auto weight : getWeights()) {
//     weight->toDevice(dev_type);
//   }
//   return BaseLayer::toDevice(dev_type);
// }

}  // namespace ginfer::layer
