#include <glog/logging.h>
#include "ginfer/op/kernels/gemm_kernel.h"
#include "ginfer/op/kernels/gemv_kernel.h"
#include "ginfer/op/kernels/kernel_registry.h"
#include "ginfer/op/op.h"

namespace ginfer::op {

MatmulOp::MatmulOp(DeviceType dev_type) : Op(dev_type, OpType::kOpMatmul) {}

Status MatmulOp::run(const std::vector<const Tensor*>& inputs, std::vector<Tensor*> outputs) {
  CHECK(inputs.size() == 2) << "MatmulOp requires exactly 2 input tensors.";
  CHECK(outputs.size() == 1) << "MatmulOp requires exactly 1 output tensor.";
  const Tensor* A = inputs[0];
  const Tensor* B = inputs[1];
  CHECK(A->dtype() == B->dtype()) << "Input tensors must have the same data type.";

  common::DeviceType dev_type = getDeviceType();
  auto dev_ctx = common::DeviceContext::create(dev_type);

  if (A->shape().ndim() == 1) {
    auto gemv_kernel =
        kernel::KernelRegistry::getInstance(dev_type)->getKernel<kernel::GemvKernelFuncType>("gemv", A->dtype());
    gemv_kernel(*dev_ctx, *A, *B, *outputs[0]);
  } else {
    auto gemm_kernel =
        kernel::KernelRegistry::getInstance(dev_type)->getKernel<kernel::GemmKernelFuncType>("gemm", A->dtype());
    gemm_kernel(*dev_ctx, *A, *B, *outputs[0]);
  }

  return ginfer::error::Success();
}

}  // namespace ginfer::op