#include "ginfer/op/kernels/embedding_kernel.h"
#include "ginfer/op/kernels/kernel_registry.h"
#include "ginfer/op/layer.h"

namespace ginfer::op {

EmbeddingLayer::EmbeddingLayer(DeviceType dev_type, std::string layer_name)
    : LayerWithParam(dev_type, LayerType::kLayerEmbedding, std::move(layer_name)) {
  resetWeightSize(1);  // embedding weight table
}

Status EmbeddingLayer::forward(const std::vector<const Tensor*>& inputs, Tensor* output) {
  CHECK(inputs.size() == 1) << "EmbeddingLayer requires exactly 1 input tensor.";

  const Tensor* input = inputs[0];
  std::shared_ptr<Tensor> weight = getWeight(0);
  CHECK(weight != nullptr) << "EmbeddingLayer weight is not set.";

  common::DeviceType dev_type = getDeviceType();

  auto kernel = kernel::KernelRegistry::getInstance(dev_type)
                    ->getKernel<kernel::EmbeddingKernelFuncType>("embedding", weight->dtype());
  auto dev_ctx = common::DeviceContext::create(dev_type);
  kernel(*dev_ctx, *input, *weight, *output);

  return ginfer::error::Success();
}

}  // namespace ginfer::op
