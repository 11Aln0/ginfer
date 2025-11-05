#include "ginfer/op/kernels/kernel_registry.h"
#include "ginfer/op/kernels/rmsnorm_kernel.h"
#include "ginfer/op/layer.h"

namespace ginfer::op {

RMSNormLayer::RMSNormLayer(DeviceType dev_type, std::string layer_name, float epsilon)
    : LayerWithParam(dev_type, LayerType::kLayerRMSNorm, std::move(layer_name)), epsilon_(epsilon) {
  resetWeightSize(1);  // gamma
}

Status RMSNormLayer::forward(const std::vector<const Tensor*>& inputs, Tensor* output) {
  CHECK(inputs.size() == 1) << "RMSNormLayer requires exactly 1 input tensor.";

  const Tensor* input = inputs[0];
  std::shared_ptr<Tensor> gamma = getWeight(0);
  CHECK(gamma != nullptr) << "RMSNormLayer gamma weight is not set.";

  // Check dimensions
  const auto& shape = input->shape();
  const auto& gamma_shape = gamma->shape();
  size_t hidden_dim = shape[shape.ndim() - 1];
  CHECK(gamma_shape.ndim() == 1 && gamma_shape[0] == hidden_dim) << "Gamma tensor shape is invalid.";

  common::DeviceType dev_type = getDeviceType();

  auto kernel = kernel::KernelRegistry::getInstance(dev_type)->getKernel<kernel::RMSNormKernelFuncType>("rmsNorm", input->dtype());
  auto dev_ctx = common::DeviceContext::create(dev_type);
  kernel(*dev_ctx, *input, *gamma, *output, epsilon_);

  return ginfer::error::Success();
}

}  // namespace ginfer::op