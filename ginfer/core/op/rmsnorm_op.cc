#include "ginfer/core/op/kernels/kernel_registry.h"
#include "ginfer/core/op/kernels/kernels.h"
#include "ginfer/core/op/op.h"

namespace ginfer::core::op {

RMSNormOp::RMSNormOp(DeviceType dev_type, float epsilon)
    : AutoKernelDispatchOp<kernel::RMSNormKernelFuncType>(dev_type, OpType::kOpRMSNorm, "rmsNorm"),
      epsilon_(epsilon) {}

Result<void, std::string> RMSNormOp::run(const core::InferContext& ctx,
                                         const std::vector<const Tensor*>& inputs,
                                         std::vector<Tensor*> outputs) {
  CHECK(inputs.size() == 2) << "RMSNormOp requires exactly 2 input tensors.";
  CHECK(outputs.size() == 1) << "RMSNormOp requires exactly 1 output tensor.";

  const Tensor* input = inputs[0];
  const Tensor* gamma = inputs[1];

  // Check dimensions
  const auto& shape = input->shape();
  const auto& gamma_shape = gamma->shape();
  size_t hidden_dim = shape[shape.ndim() - 1];
  CHECK(gamma_shape.ndim() == 1 && gamma_shape[0] == hidden_dim)
      << "Gamma tensor shape is invalid.";

  common::DeviceType dev_type = getDeviceType();

  auto kernel = getKernel(dev_type, input->dtype());
  const auto& dev_ctx = getDeviceContext(ctx);
  kernel(dev_ctx, *input, *gamma, *outputs[0], epsilon_);

  return Ok<void>();
}

}  // namespace ginfer::core::op