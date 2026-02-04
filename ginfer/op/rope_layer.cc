#include "ginfer/op/kernels/kernel_registry.h"
#include "ginfer/op/kernels/rope_kernel.h"
#include "ginfer/op/layer.h"

namespace ginfer::op {

ROPELayer::ROPELayer(DeviceType dev_type, std::string layer_name, int head_dim, int max_seq_len, float rope_theta)
    : Layer(dev_type, LayerType::kLayerROPE, std::move(layer_name)),
      head_dim_(head_dim),
      rope_theta_(rope_theta),
      max_seq_len_(max_seq_len) {
  sin_cache_ = std::make_shared<Tensor>(tensor::DataType::kDataTypeFloat32, tensor::Shape({max_seq_len, head_dim_ / 2}),
                                        dev_type);
  cos_cache_ = std::make_shared<Tensor>(tensor::DataType::kDataTypeFloat32, tensor::Shape({max_seq_len, head_dim_ / 2}),
                                        dev_type);
}

void ROPELayer::updateCache(int start_pos, int end_pos) {
  CHECK(start_pos >= 0 && end_pos <= max_seq_len_ && start_pos < end_pos)
      << "ROPELayer cache update failed: invalid start or end positions.";

  common::DeviceType dev_type = getDeviceType();

  auto kernel = kernel::KernelRegistry::getInstance(dev_type)->getKernel<kernel::CalcSinCosKernelFuncType>(
      "calcSinCos", tensor::DataType::kDataTypeFloat32);
  auto dev_ctx = common::DeviceContext::create(dev_type);
  kernel(*dev_ctx, *sin_cache_, *cos_cache_, start_pos, end_pos, head_dim_, rope_theta_);
}

Status ROPELayer::forward(const std::vector<const Tensor*>& inputs, Tensor* output) {
  CHECK(inputs.size() == 1) << "ROPELayer requires exactly 1 input tensor.";

  const Tensor* input = inputs[0];

  CHECK(sin_cache_ != nullptr && cos_cache_ != nullptr)
      << "ROPELayer sin/cos cache not initialized. Call updateCache() first.";

  common::DeviceType dev_type = getDeviceType();

  auto kernel =
      kernel::KernelRegistry::getInstance(dev_type)->getKernel<kernel::ROPEKernelFuncType>("ROPE", input->dtype());
  auto dev_ctx = common::DeviceContext::create(dev_type);
  kernel(*dev_ctx, *input, *output, *sin_cache_, *cos_cache_);

  return ginfer::error::Success();
}

}  // namespace ginfer::op
