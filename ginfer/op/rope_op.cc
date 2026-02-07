#include "ginfer/op/kernels/kernel_registry.h"
#include "ginfer/op/kernels/rope_kernel.h"
#include "ginfer/op/op.h"

namespace ginfer::op {

ROPESinCosCacheOp::ROPESinCosCacheOp(DeviceType dev_type, int head_dim, int max_seq_len, float rope_theta)
    : Op(dev_type, OpType::kOpCustom, "rope_sin_cos_cache"),
      head_dim_(head_dim),
      rope_theta_(rope_theta),
      max_seq_len_(max_seq_len) {}

Status ROPESinCosCacheOp::run(const std::vector<const Tensor*>& inputs, std::vector<Tensor*> outputs) {
  CHECK(inputs.empty()) << "ROPESinCosCacheOp does not take any input tensors.";
  CHECK(outputs.size() == 2) << "ROPESinCosCacheOp requires exactly 2 output tensors.";

  common::DeviceType dev_type = getDeviceType();

  auto kernel = kernel::KernelRegistry::getInstance(dev_type)->getKernel<kernel::CalcSinCosKernelFuncType>(
      "calcSinCos", tensor::DataType::kDataTypeFloat32);
  auto dev_ctx = common::DeviceContext::create(dev_type);
  kernel(*dev_ctx, *outputs[0], *outputs[1], 0, max_seq_len_, head_dim_, rope_theta_);

  return ginfer::error::Success();
}

ROPEOp::ROPEOp(DeviceType dev_type, int head_dim, int max_seq_len, float rope_theta)
    : Op(dev_type, OpType::kOpROPE, "rope"), head_dim_(head_dim), rope_theta_(rope_theta), max_seq_len_(max_seq_len) {}

Status ROPEOp::run(const std::vector<const Tensor*>& inputs, std::vector<Tensor*> outputs) {
  CHECK(inputs.size() == 3) << "ROPEOp requires exactly 3 input tensors.";
  CHECK(outputs.size() == 1) << "ROPEOp requires exactly 1 output tensor.";

  const Tensor* input = inputs[0];
  const Tensor* sin_cache = inputs[1];
  const Tensor* cos_cache = inputs[2];

  common::DeviceType dev_type = getDeviceType();

  auto kernel =
      kernel::KernelRegistry::getInstance(dev_type)->getKernel<kernel::ROPEKernelFuncType>("ROPE", input->dtype());
  auto dev_ctx = common::DeviceContext::create(dev_type);
  kernel(*dev_ctx, *input, *outputs[0], *sin_cache, *cos_cache);

  return ginfer::error::Success();
}

}  // namespace ginfer::op
