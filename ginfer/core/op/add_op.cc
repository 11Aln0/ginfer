#include "ginfer/core/op/kernels/kernel_registry.h"
#include "ginfer/core/op/kernels/kernels.h"
#include "ginfer/core/op/op.h"

namespace ginfer::core::op {

AddOp::AddOp(DeviceType dev_type) : Op(dev_type, OpType::kOpAdd, "add") {}

Result<void, std::string> AddOp::run(const core::InferContext& ctx,
                                     const std::vector<const Tensor*>& inputs,
                                     std::vector<Tensor*> outputs) {
  CHECK(inputs.size() == 2) << "AddOp requires exactly 2 input tensors.";
  CHECK(outputs.size() == 1) << "AddOp requires exactly 1 output tensor.";

  // TODO broadcast add

  tensor::DataType dtype = inputs[0]->dtype();
  CHECK(dtype == inputs[1]->dtype() && dtype == outputs[0]->dtype())
      << "Input tensors must have the same data type.";

  common::DeviceType dev_type = getDeviceType();

  auto kernel = kernel::KernelRegistry::getInstance(dev_type)->getKernel<kernel::AddKernelFuncType>(
      "add", dtype);
  auto dev_ctx = common::DeviceContext::create(dev_type);
  kernel(*dev_ctx, *inputs[0], *inputs[1], *outputs[0]);

  return Ok<void>();
}

};  // namespace ginfer::core::op