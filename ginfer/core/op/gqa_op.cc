#include <glog/logging.h>
#include "ginfer/core/op/kernels/gqa_kernel.h"
#include "ginfer/core/op/kernels/kernel_registry.h"
#include "ginfer/core/op/op.h"

namespace ginfer::core::op {

GQAOp::GQAOp(DeviceType dev_type) : Op(dev_type, OpType::kOpGQA, "gqa") {}

Result<void, std::string> GQAOp::run(const common::InferContext& ctx,
                                     const std::vector<const Tensor*>& inputs,
                                     std::vector<Tensor*> outputs) {
  CHECK(inputs.size() == 3) << "GQAOp requires exactly 3 input tensors.";
  CHECK(outputs.size() == 1) << "GQAOp requires exactly 1 output tensor.";
  const Tensor* q = inputs[0];
  const Tensor* k = inputs[1];
  const Tensor* v = inputs[2];
  CHECK(q->dtype() == k->dtype() && k->dtype() == v->dtype())
      << "Input tensors must have the same data type.";

  common::DeviceType dev_type = getDeviceType();
  auto dev_ctx = common::DeviceContext::create(dev_type);

  auto gqa_kernel =
      kernel::KernelRegistry::getInstance(dev_type)->getKernel<kernel::GQAKernelFuncType>(
          "GQA", q->dtype());
  gqa_kernel(*dev_ctx, *q, *k, *v, *outputs[0]);

  return Ok<void>();
}

// void GQAOp::setSeqLen(int seq_len) { seq_len_ = seq_len; }

}  // namespace ginfer::core::op