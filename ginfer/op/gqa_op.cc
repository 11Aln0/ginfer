#include <glog/logging.h>
#include "ginfer/op/kernels/gqa_kernel.h"
#include "ginfer/op/kernels/kernel_registry.h"
#include "ginfer/op/op.h"

namespace ginfer::op {

GQAOp::GQAOp(DeviceType dev_type) : Op(dev_type, OpType::kOpGQA, "gqa") {}

Result<void, std::string> GQAOp::run(const common::InferContext& ctx, const std::vector<const Tensor*>& inputs,
                                     std::vector<Tensor*> outputs) {
  CHECK(inputs.size() == 3) << "GQAOp requires exactly 3 input tensors.";
  CHECK(outputs.size() == 1) << "GQAOp requires exactly 1 output tensor.";
  const Tensor* q = inputs[0];
  const Tensor* k = inputs[1];
  const Tensor* v = inputs[2];
  CHECK(q->dtype() == k->dtype() && k->dtype() == v->dtype()) << "Input tensors must have the same data type.";

  common::DeviceType dev_type = getDeviceType();
  auto dev_ctx = common::DeviceContext::create(dev_type);

  auto gqa_kernel =
      kernel::KernelRegistry::getInstance(dev_type)->getKernel<kernel::GQAKernelFuncType>("GQA", q->dtype());
  gqa_kernel(*dev_ctx, *q, *k, *v, *outputs[0]);

  return Ok<void>();
}

// void GQAOp::setSeqLen(int seq_len) { seq_len_ = seq_len; }

GQAVarlenOp::GQAVarlenOp(DeviceType dev_type, int paged_block_size)
    : Op(dev_type, OpType::kOpGQA, "gqa_varlen"), paged_block_size_(paged_block_size) {}

Result<void, std::string> GQAVarlenOp::run(const common::InferContext& ctx, const std::vector<const Tensor*>& inputs,
                                           std::vector<Tensor*> outputs) {
  CHECK(inputs.size() == 6) << "GQAVarlenOp requires exactly 6 input tensors.";
  CHECK(outputs.size() == 1) << "GQAVarlenOp requires exactly 1 output tensor.";
  const Tensor* q = inputs[0];
  const Tensor* k = inputs[1];
  const Tensor* v = inputs[2];
  const Tensor* cu_seqlens_q = inputs[3];
  const Tensor* cu_seqlens_kv = inputs[4];
  const Tensor* block_tables = inputs[5];
  CHECK(q->dtype() == k->dtype() && k->dtype() == v->dtype()) << "Input tensors must have the same data type.";
  CHECK(ctx.max_seqlen_q.has_value()) << "GQAVarlenOp requires max_seqlen_q in InferContext.";
  int max_seqlen_q = ctx.max_seqlen_q.value();

  common::DeviceType dev_type = getDeviceType();
  auto dev_ctx = common::DeviceContext::create(dev_type);

  auto gqa_varlen_kernel = kernel::KernelRegistry::getInstance(dev_type)->getKernel<kernel::GQAVarlenKernelFuncType>(
      "GQAVarlen", q->dtype());
  gqa_varlen_kernel(*dev_ctx, *q, *k, *v, *cu_seqlens_q, *cu_seqlens_kv, *block_tables, max_seqlen_q, paged_block_size_,
                    *outputs[0]);

  return Ok<void>();
}

}  // namespace ginfer::op