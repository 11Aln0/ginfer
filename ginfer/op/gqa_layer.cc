#include <glog/logging.h>
#include "ginfer/op/kernels/gqa_kernel.h"
#include "ginfer/op/kernels/kernel_registry.h"
#include "ginfer/op/layer.h"

namespace ginfer::op {

GQALayer::GQALayer(DeviceType dev_type, std::string layer_name)
    : Layer(dev_type, LayerType::kLayerGQA, std::move(layer_name)), seq_len_(0) {}

Status GQALayer::forward(const std::vector<const Tensor*>& inputs, Tensor* output) {
  CHECK(inputs.size() == 3);
  const Tensor* q = inputs[0];
  const Tensor* k = inputs[1];
  const Tensor* v = inputs[2];
  CHECK(q->dtype() == k->dtype() && k->dtype() == v->dtype()) << "Input tensors must have the same data type.";

  common::DeviceType dev_type = getDeviceType();
  auto dev_ctx = common::DeviceContext::create(dev_type);

  auto gqa_kernel =
      kernel::KernelRegistry::getInstance(dev_type)->getKernel<kernel::GQAKernelFuncType>("GQA", q->dtype());
  gqa_kernel(*dev_ctx, *q, *k, *v, *output, seq_len_);

  return ginfer::error::Success();
}

void GQALayer::setSeqLen(int seq_len) { seq_len_ = seq_len; }

}  // namespace ginfer::op