#include <glog/logging.h>
#include <optional>
#include "ginfer/op/kernels/gemm_kernel.h"
#include "ginfer/op/kernels/gemv_kernel.h"
#include "ginfer/op/kernels/kernel_registry.h"
#include "ginfer/op/op.h"

namespace ginfer::op {

MatmulOp::MatmulOp(DeviceType dev_type) : Op(dev_type, OpType::kOpMatmul, "matmul") {}

Result<void, std::string> MatmulOp::run(const std::vector<const Tensor*>& inputs, std::vector<Tensor*> outputs) {
  CHECK(inputs.size() == 2 || inputs.size() == 3) << "MatmulOp requires exactly 2 or 3 input tensors.";
  CHECK(outputs.size() == 1) << "MatmulOp requires exactly 1 output tensor.";
  const Tensor* A = inputs[0];
  const Tensor* B = inputs[1];
  const Tensor* bias = nullptr;
  if (inputs.size() == 3) {
    bias = inputs[2];
  }
  CHECK(A->dtype() == B->dtype()) << "Input tensors must have the same data type.";

  common::DeviceType dev_type = getDeviceType();
  auto dev_ctx = common::DeviceContext::create(dev_type);

  if (isGemvMode(A)) {
    auto gemv_kernel =
        kernel::KernelRegistry::getInstance(dev_type)->getKernel<kernel::GemvKernelFuncType>("gemv", A->dtype());
    if (bias != nullptr) {
      gemv_kernel(*dev_ctx, *A, *B, *bias, *outputs[0]);
    } else {
      gemv_kernel(*dev_ctx, *A, *B, std::nullopt, *outputs[0]);
    }
  } else {
    auto gemm_kernel =
        kernel::KernelRegistry::getInstance(dev_type)->getKernel<kernel::GemmKernelFuncType>("gemm", A->dtype());
    if (bias != nullptr) {
      gemm_kernel(*dev_ctx, *A, *B, *bias, *outputs[0]);
    } else {
      gemm_kernel(*dev_ctx, *A, *B, std::nullopt, *outputs[0]);
    }
  }

  return Ok<void>();
}

bool MatmulOp::isGemvMode(const Tensor* A) {
  return std::accumulate(A->shape().begin(), A->shape().end(), 1, std::multiplies<size_t>()) == A->shape()[0];
}

}  // namespace ginfer::op