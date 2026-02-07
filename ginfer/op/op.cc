#include "ginfer/op/op.h"
#include <glog/logging.h>
#include "ginfer/common/errors.h"

namespace ginfer::op {

BaseOp::BaseOp(DeviceType dev_type, OpType op_type, std::string name)
    : dev_type_(dev_type), op_type_(op_type), name_(std::move(name)) {}

OpType BaseOp::opType() const { return op_type_; }

DeviceType BaseOp::getDeviceType() const { return dev_type_; }

};  // namespace ginfer::op
