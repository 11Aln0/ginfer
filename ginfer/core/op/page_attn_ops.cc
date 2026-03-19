
#include <glog/logging.h>
#include "ginfer/core/op/kernels/gqa_kernel.h"
#include "ginfer/core/op/kernels/kernel_registry.h"
#include "ginfer/core/op/kernels/page_attn_kernel.h"
#include "ginfer/core/op/op.h"

namespace ginfer::core::op {

GQAVarlenOp::GQAVarlenOp(DeviceType dev_type, int paged_block_size)
    : Op(dev_type, OpType::kOpGQA, "gqa_varlen"), paged_block_size_(paged_block_size) {}

Result<void, std::string> GQAVarlenOp::run(const core::InferContext& ctx,
                                           const std::vector<const Tensor*>& inputs,
                                           std::vector<Tensor*> outputs) {
  CHECK(inputs.size() == 6) << "GQAVarlenOp requires exactly 6 input tensors.";
  CHECK(outputs.size() == 1) << "GQAVarlenOp requires exactly 1 output tensor.";
  const Tensor* q = inputs[0];
  const Tensor* k = inputs[1];
  const Tensor* v = inputs[2];
  const Tensor* cu_seqlens_q = inputs[3];
  const Tensor* cu_seqlens_kv = inputs[4];
  const Tensor* block_tables = inputs[5];
  CHECK(q->dtype() == k->dtype() && k->dtype() == v->dtype())
      << "Input tensors must have the same data type.";
  CHECK(ctx.max_seqlen_q.has_value()) << "GQAVarlenOp requires max_seqlen_q in InferContext.";
  int max_seqlen_q = ctx.max_seqlen_q.value();

  common::DeviceType dev_type = getDeviceType();
  auto dev_ctx = common::DeviceContext::create(dev_type);

  auto gqa_varlen_kernel =
      kernel::KernelRegistry::getInstance(dev_type)->getKernel<kernel::GQAVarlenKernelFuncType>(
          "GQAVarlen", q->dtype());
  gqa_varlen_kernel(*dev_ctx, *q, *k, *v, *cu_seqlens_q, *cu_seqlens_kv, *block_tables,
                    max_seqlen_q, paged_block_size_, *outputs[0]);

  return Ok<void>();
}

StoreKVCacheOp::StoreKVCacheOp(DeviceType dev_type)
    : Op(dev_type, OpType::kOpCustom, "store_kvcache") {}

Result<void, std::string> StoreKVCacheOp::run(const core::InferContext& ctx,
                                              const std::vector<const Tensor*>& inputs,
                                              std::vector<Tensor*> outputs) {
  // inputs: [k, v, k_cache, v_cache, slot_mapping]  — all in-place, no outputs
  CHECK(inputs.size() == 5)
      << "StoreKVCacheOp requires exactly 5 input tensors (k, v, k_cache, v_cache, slot_mapping).";
  CHECK(outputs.empty()) << "StoreKVCacheOp has no outputs (in-place op).";

  const Tensor* k = inputs[0];
  const Tensor* v = inputs[1];
  Tensor* k_cache = const_cast<Tensor*>(inputs[2]);
  Tensor* v_cache = const_cast<Tensor*>(inputs[3]);
  const Tensor* slot_mapping = inputs[4];

  CHECK(k->dtype() == v->dtype()) << "k and v must have the same data type.";
  CHECK(k->dtype() == k_cache->dtype()) << "k and k_cache must have the same data type.";

  common::DeviceType dev_type = getDeviceType();
  auto dev_ctx = common::DeviceContext::create(dev_type);

  auto kernel =
      kernel::KernelRegistry::getInstance(dev_type)->getKernel<kernel::StoreKVCacheKernelFuncType>(
          "storeKVCache", k->dtype());
  kernel(*dev_ctx, *k, *v, *k_cache, *v_cache, *slot_mapping);

  return Ok<void>();
}

}  // namespace ginfer::core::op