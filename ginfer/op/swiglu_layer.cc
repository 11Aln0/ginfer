#include "ginfer/op/kernels/kernel_registry.h"
#include "ginfer/op/kernels/swiglu_kernel.h"
#include "ginfer/op/layer.h"

namespace ginfer::op {

SwiGLULayer::SwiGLULayer(DeviceType dev_type, std::string layer_name)
    : Layer(dev_type, LayerType::kLayerSwiGLU, std::move(layer_name)) {}

Status SwiGLULayer::forward(const std::vector<const Tensor*>& inputs, Tensor* output) {
  CHECK(inputs.size() == 2) << "SwiGLULayer requires exactly 2 input tensors (gate, up).";

  const Tensor* gate = inputs[0];
  const Tensor* up = inputs[1];

  common::DeviceType dev_type = getDeviceType();

  auto kernel = kernel::KernelRegistry::getInstance(dev_type)
                    ->getKernel<kernel::SwiGluKernelFuncType>("swiglu", gate->dtype());
  auto dev_ctx = common::DeviceContext::create(dev_type);
  kernel(*dev_ctx, *output, *gate, *up);

  return ginfer::error::Success();
}

}  // namespace ginfer::op
