#include "ginfer/core/op/kernels/kernel_registry.h"
#include "ginfer/core/op/kernels/kernels.h"
#include "ginfer/core/op/op.h"

namespace ginfer::core::op {

SelectLastTokenOp::SelectLastTokenOp(DeviceType dev_type)
    : AutoKernelDispatchOp<kernel::SelectLastTokenKernelFuncType>(dev_type, OpType::kOpCustom,
                                                                  "select_last_token",
                                                                  "selectLastToken") {}

Result<void, std::string> SelectLastTokenOp::run(const core::InferContext& ctx,
                                                 const std::vector<const Tensor*>& inputs,
                                                 std::vector<Tensor*> outputs) {
  CHECK(inputs.size() == 2)
      << "SelectLastTokenOp requires exactly 2 input tensors (input, cu_seqlen_q).";
  CHECK(outputs.size() == 1) << "SelectLastTokenOp requires exactly 1 output tensor.";

  const Tensor* input = inputs[0];
  const Tensor* cu_seqlen_q = inputs[1];
  Tensor* output = outputs[0];

  CHECK(input->dtype() == output->dtype()) << "Input and output tensors must have the same dtype.";
  CHECK(cu_seqlen_q->dtype() == tensor::DataType::kDataTypeInt32)
      << "cu_seqlen_q dtype must be int32.";

  common::DeviceType dev_type = getDeviceType();

  auto kernel = getKernel(dev_type, input->dtype());
  const auto& dev_ctx = getDeviceContext(ctx);
  kernel(dev_ctx, *input, *cu_seqlen_q, *output);

  return Ok<void>();
}

}  // namespace ginfer::core::op
