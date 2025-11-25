#include <fmt/core.h>
#include "ginfer/op/kernels/add_kernel.h"
#include "ginfer/op/kernels/kernel_registry.h"
#include "ginfer/op/layer.h"

namespace ginfer::op {

AddLayer::AddLayer(DeviceType dev_type, std::string layer_name)
    : Layer(dev_type, LayerType::kLayerAdd, std::move(layer_name)) {}

Status AddLayer::forward(const std::vector<const Tensor*>& inputs, Tensor* output) {
  CHECK(inputs.size() == 2) << "AddLayer requires exactly 2 input tensors.";

  // TODO broadcast add
  tensor::DataType dtype = inputs[0]->dtype();
  CHECK(dtype == inputs[1]->dtype() && dtype == output->dtype()) << "Input tensors must have the same data type.";

  common::DeviceType dev_type = getDeviceType();

  auto kernel = kernel::KernelRegistry::getInstance(dev_type)->getKernel<kernel::AddKernelFuncType>("add", dtype);
  auto dev_ctx = common::DeviceContext::create(dev_type);
  kernel(*dev_ctx, *inputs[0], *inputs[1], *output);

  return ginfer::error::Success();
}

};  // namespace ginfer::op