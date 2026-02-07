#include "ginfer/op/kernels/embedding_kernel.h"
#include "ginfer/op/kernels/kernel_registry.h"
#include "ginfer/op/op.h"

namespace ginfer::op {

EmbeddingOp::EmbeddingOp(DeviceType dev_type) : Op(dev_type, OpType::kOpEmbedding) {}

Status EmbeddingOp::run(const std::vector<const Tensor*>& inputs, std::vector<Tensor*> outputs) {
  CHECK(inputs.size() == 2) << "EmbeddingOp requires exactly 2 input tensors.";
  CHECK(outputs.size() == 1) << "EmbeddingOp requires exactly 1 output tensor.";

  const Tensor* input = inputs[0];
  const Tensor* weight = inputs[1];

  common::DeviceType dev_type = getDeviceType();

  auto kernel = kernel::KernelRegistry::getInstance(dev_type)->getKernel<kernel::EmbeddingKernelFuncType>(
      "embedding", weight->dtype());
  auto dev_ctx = common::DeviceContext::create(dev_type);
  kernel(*dev_ctx, *input, *weight, *outputs[0]);

  return ginfer::error::Success();
}

}  // namespace ginfer::op
