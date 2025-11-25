#include <glog/logging.h>
#include "ginfer/op/kernels/gemv_kernel.h"
#include "ginfer/op/kernels/kernel_registry.h"
#include "ginfer/op/layer.h"

namespace ginfer::op {

MatmulLayer::MatmulLayer(DeviceType dev_type, std::string layer_name)
    : Layer(dev_type, LayerType::kLayerMatmul, std::move(layer_name)) {}

Status MatmulLayer::forward(const std::vector<const Tensor*>& inputs, Tensor* output) {
  CHECK(inputs.size() == 2);
  const Tensor* A = inputs[0];
  const Tensor* B = inputs[1];
  CHECK(A->dtype() == B->dtype()) << "Input tensors must have the same data type.";

  common::DeviceType dev_type = getDeviceType();
  auto dev_ctx = common::DeviceContext::create(dev_type);

  if (A->shape().ndim() == 1) {
    auto gemv_kernel =
        kernel::KernelRegistry::getInstance(dev_type)->getKernel<kernel::GemvKernelFuncType>("gemv", A->dtype());
    gemv_kernel(*dev_ctx, *B, *A, *output);
  } else {
    LOG(FATAL) << "MatmulLayer only supports Gemv (A is 1-D tensor) for now, got A ndim = " << A->shape().ndim();
  }

  return ginfer::error::Success();
}

}  // namespace ginfer::op