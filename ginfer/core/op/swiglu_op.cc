#include "ginfer/core/op/kernels/kernel_registry.h"
#include "ginfer/core/op/kernels/swiglu_kernel.h"
#include "ginfer/core/op/op.h"

namespace ginfer::core::op {

SwiGLUOp::SwiGLUOp(DeviceType dev_type) : Op(dev_type, OpType::kOpSwiGLU, "swiglu") {}

Result<void, std::string> SwiGLUOp::run(const core::InferContext& ctx,
                                        const std::vector<const Tensor*>& inputs,
                                        std::vector<Tensor*> outputs) {
  CHECK(inputs.size() == 2) << "SwiGLUOp requires exactly 2 input tensors (gate, up).";
  CHECK(outputs.size() == 1) << "SwiGLUOp requires exactly 1 output tensor.";

  const Tensor* gate = inputs[0];
  const Tensor* up = inputs[1];

  common::DeviceType dev_type = getDeviceType();

  auto kernel =
      kernel::KernelRegistry::getInstance(dev_type)->getKernel<kernel::SwiGluKernelFuncType>(
          "swiglu", gate->dtype());
  auto dev_ctx = common::DeviceContext::create(dev_type);
  kernel(*dev_ctx, *outputs[0], *gate, *up);

  return Ok<void>();
}

}  // namespace ginfer::core::op
