#include "ginfer/op/kernels/argmax_kernel.h"
#include "ginfer/op/kernels/kernel_registry.h"
#include "ginfer/op/layer.h"

namespace ginfer::op {

ArgmaxLayer::ArgmaxLayer(DeviceType dev_type, std::string layer_name)
    : Layer(dev_type, LayerType::kLayerArgmax, std::move(layer_name)) {}

Status ArgmaxLayer::forward(const std::vector<const Tensor*>& inputs, Tensor* output) {
  CHECK(inputs.size() == 1) << "ArgmaxLayer requires exactly 1 input tensor.";

  const Tensor* input = inputs[0];

  common::DeviceType dev_type = getDeviceType();

  auto kernel = kernel::KernelRegistry::getInstance(dev_type)
                    ->getKernel<kernel::ArgmaxKernelFuncType>("argmax", input->dtype());
  auto dev_ctx = common::DeviceContext::create(dev_type);
  kernel(*dev_ctx, *input, *output);

  return ginfer::error::Success();
}

}  // namespace ginfer::op
