#include "ginfer/op/kernels/add_kernel.h"
#include "ginfer/op/kernels/kernel_registry.h"
#include "ginfer/op/layer.h"

namespace ginfer::op {
AddLayer::AddLayer(DeviceType dev_type, std::string layer_name)
    : Layer(dev_type, LayerType::kLayerAdd, std::move(layer_name)) {}

Status AddLayer::forward(const std::vector<const Tensor*>& inputs, Tensor* output) {
  CHECK(inputs.size() == 2) << "AddLayer requires exactly 2 input tensors.";

  auto kernel =
      kernel::KernelRegistry::getInstance(devType())->getKernel<kernel::AddKernelFuncType>(
          kernel::KernelInfo("add", inputs[0]->dtype(), output->dtype(), devType()));
  return ginfer::error::Success();
}

};  // namespace ginfer::op