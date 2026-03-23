#include "ginfer/core/op/kernels/kernel_dispatcher.h"
#include "ginfer/core/op/kernels/kernel_registry.h"
#include "ginfer/core/op/kernels/kernels.h"
#include "ginfer/core/op/op.h"

namespace ginfer::core::op {

ArgmaxOp::ArgmaxOp(DeviceType dev_type)
    : AutoKernelDispatchOp<kernel::ArgmaxKernelFuncType>(dev_type, OpType::kOpArgmax, "argmax") {}

Result<void, std::string> ArgmaxOp::run(const core::InferContext& ctx,
                                        const std::vector<const Tensor*>& inputs,
                                        std::vector<Tensor*> outputs) {
  CHECK(inputs.size() == 1) << "ArgmaxOp requires exactly 1 input tensor.";
  CHECK(outputs.size() == 1) << "ArgmaxOp requires exactly 1 output tensor.";

  const Tensor* input = inputs[0];
  Tensor* output = outputs[0];

  common::DeviceType dev_type = getDeviceType();

  auto kernel = getKernel(dev_type, input->dtype(), output->dtype());
  auto dev_ctx = common::DeviceContext::create(dev_type);
  kernel(*dev_ctx, *input, *output);

  return Ok<void>();
}

}  // namespace ginfer::core::op
