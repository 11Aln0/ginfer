#include "ginfer/core/op/kernels/embedding_kernel.h"
#include "ginfer/core/op/kernels/kernel_registry.h"
#include "ginfer/core/op/op.h"

namespace ginfer::core::op {

EmbeddingOp::EmbeddingOp(DeviceType dev_type) : Op(dev_type, OpType::kOpEmbedding, "embedding") {}

Result<void, std::string> EmbeddingOp::run(const common::InferContext& ctx,
                                           const std::vector<const Tensor*>& inputs,
                                           std::vector<Tensor*> outputs) {
  CHECK(inputs.size() == 2) << "EmbeddingOp requires exactly 2 input tensors.";
  CHECK(outputs.size() == 1) << "EmbeddingOp requires exactly 1 output tensor.";

  const Tensor* input = inputs[0];
  const Tensor* weight = inputs[1];

  common::DeviceType dev_type = getDeviceType();

  auto kernel =
      kernel::KernelRegistry::getInstance(dev_type)->getKernel<kernel::EmbeddingKernelFuncType>(
          "embedding", weight->dtype());
  auto dev_ctx = common::DeviceContext::create(dev_type);
  kernel(*dev_ctx, *input, *weight, *outputs[0]);

  return Ok<void>();
}

}  // namespace ginfer::core::op
