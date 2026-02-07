#include "ginfer/op/kernels/argmax_kernel.h"
#include "ginfer/op/kernels/kernel_registry.h"
#include "ginfer/op/op.h"

namespace ginfer::op {

ArgmaxOp::ArgmaxOp(DeviceType dev_type) : Op(dev_type, OpType::kOpArgmax) {}

Status ArgmaxOp::run(const std::vector<const Tensor*>& inputs, std::vector<Tensor*> outputs) {
  CHECK(inputs.size() == 1) << "ArgmaxOp requires exactly 1 input tensor.";
  CHECK(outputs.size() == 1) << "ArgmaxOp requires exactly 1 output tensor.";

  const Tensor* input = inputs[0];

  common::DeviceType dev_type = getDeviceType();

  auto kernel =
      kernel::KernelRegistry::getInstance(dev_type)->getKernel<kernel::ArgmaxKernelFuncType>("argmax", input->dtype());
  auto dev_ctx = common::DeviceContext::create(dev_type);
  kernel(*dev_ctx, *input, *outputs[0]);

  return ginfer::error::Success();
}

}  // namespace ginfer::op
