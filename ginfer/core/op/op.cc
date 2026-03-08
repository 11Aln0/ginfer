#include "ginfer/core/op/op.h"
#include <glog/logging.h>
#include "ginfer/common/errors.h"

namespace ginfer::core::op {

BaseOp::BaseOp(DeviceType dev_type, OpType op_type, std::string name)
    : dev_type_(dev_type), op_type_(op_type), name_(std::move(name)) {}

Result<void, std::string> BaseOp::toDevice(DeviceType dev_type) {
  dev_type_ = dev_type;
  return Ok<void>();
}

OpType BaseOp::opType() const { return op_type_; }

DeviceType BaseOp::getDeviceType() const { return dev_type_; }

};  // namespace ginfer::core::op
