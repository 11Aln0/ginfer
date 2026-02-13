#include "ginfer/op/kernels/kernel_registry.h"
#include "ginfer/op/kernels/rope_kernel.h"
#include "ginfer/op/op.h"

namespace ginfer::op {

RotaryEmbeddingOp::RotaryEmbeddingOp(DeviceType dev_type, float rope_theta)
    : Op(dev_type, OpType::kOpCustom, "rotary_embedding"), rope_theta_(rope_theta) {}

Status RotaryEmbeddingOp::run(const std::vector<const Tensor*>& inputs, std::vector<Tensor*> outputs) {
  CHECK(inputs.size() == 1) << "RotaryEmbeddingOp requires exactly 1 input tensor.";
  CHECK(outputs.size() == 2) << "RotaryEmbeddingOp requires exactly 2 output tensors.";

  common::DeviceType dev_type = getDeviceType();
  auto pos_ids_range = inputs[0]->data<int64_t>();

  auto kernel = kernel::KernelRegistry::getInstance(dev_type)->getKernel<kernel::RotaryEmbeddingKernelFuncType>(
      "rotary_embedding", tensor::DataType::kDataTypeFloat32);
  auto dev_ctx = common::DeviceContext::create(dev_type);
  kernel(*dev_ctx, *outputs[0], *outputs[1], pos_ids_range[0], pos_ids_range[1], rope_theta_);

  return ginfer::error::Success();
}

ROPEOp::ROPEOp(DeviceType dev_type) : Op(dev_type, OpType::kOpROPE, "rope") {}

Status ROPEOp::run(const std::vector<const Tensor*>& inputs, std::vector<Tensor*> outputs) {
  CHECK(inputs.size() == 3) << "ROPEOp requires exactly 3 input tensors.";
  CHECK(outputs.size() == 1) << "ROPEOp requires exactly 1 output tensor.";

  const Tensor* input = inputs[0];  // [seq_len, num_heads or num_kv_heads, head_dim]
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
