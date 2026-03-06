#include "ginfer/op/kernels/kernel_registry.h"
#include "ginfer/op/kernels/swiglu_kernel.h"
#include "ginfer/op/op.h"

namespace ginfer::op {

SwiGLUOp::SwiGLUOp(DeviceType dev_type) : Op(dev_type, OpType::kOpSwiGLU, "swiglu") {}

Result<void, std::string> SwiGLUOp::run(const common::InferContext& ctx, const std::vector<const Tensor*>& inputs,
                                        std::vector<Tensor*> outputs) {
  CHECK(inputs.size() == 2) << "SwiGLUOp requires exactly 2 input tensors (gate, up).";
  CHECK(outputs.size() == 1) << "SwiGLUOp requires exactly 1 output tensor.";

  const Tensor* gate = inputs[0];
  const Tensor* up = inputs[1];

  common::DeviceType dev_type = getDeviceType();

  auto kernel =
      kernel::KernelRegistry::getInstance(dev_type)->getKernel<kernel::SwiGluKernelFuncType>("swiglu", gate->dtype());
  auto dev_ctx = common::DeviceContext::create(dev_type);
  kernel(*dev_ctx, *outputs[0], *gate, *up);

  return Ok<void>();
}

}  // namespace ginfer::op
