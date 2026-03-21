#include <glog/logging.h>
#include <vector>
#include "ginfer/core/context.h"
#include "ginfer/core/op/op.h"
#include "ginfer/test/pybind/func_wrap.h"
#include "ginfer/test/pybind/test_registry.h"
#include "ginfer/test/pybind/type.h"

namespace py = pybind11;

namespace ginfer::test::pybind {

using common::DeviceType;
using core::tensor::DataType;
using core::tensor::Shape;
using core::tensor::Tensor;
using core::tensor::TensorRef;

// ─────────────────────────────────────────────
//  test_gqa_varlen_op_cuda
//
//  q:            [total_q_tokens, num_heads, head_dim]
//  k / v:        [total_blocks * block_size, kv_heads, head_dim]  (paged layout)
//  cu_seqlens_q: [batch + 1]  int32
//  cu_seqlens_kv:[batch + 1]  int32
//  block_tables: [batch, max_blocks_per_seq] int32
// ─────────────────────────────────────────────
TensorRef test_gqa_varlen_op_cuda(TensorRef q_tensor,
                                  TensorRef k_tensor,
                                  TensorRef v_tensor,
                                  TensorRef cu_seqlens_q_tensor,
                                  TensorRef cu_seqlens_kv_tensor,
                                  TensorRef block_tables_tensor,
                                  int max_seqlen_q,
                                  int paged_block_size) {
  auto out_res =
      Tensor::create(q_tensor->dtype(), Shape(q_tensor->shape()), DeviceType::kDeviceCPU);
  CHECK(out_res.ok()) << out_res.err();
  auto output_tensor = out_res.value();

  ::ginfer::core::op::GQAVarlenOp op(DeviceType::kDeviceCUDA, paged_block_size);

  ASSIGN_OR_THROW(q_tensor, q_tensor->toDevice(DeviceType::kDeviceCUDA));
  ASSIGN_OR_THROW(k_tensor, k_tensor->toDevice(DeviceType::kDeviceCUDA));
  ASSIGN_OR_THROW(v_tensor, v_tensor->toDevice(DeviceType::kDeviceCUDA));
  ASSIGN_OR_THROW(cu_seqlens_q_tensor, cu_seqlens_q_tensor->toDevice(DeviceType::kDeviceCUDA));
  ASSIGN_OR_THROW(cu_seqlens_kv_tensor, cu_seqlens_kv_tensor->toDevice(DeviceType::kDeviceCUDA));
  ASSIGN_OR_THROW(block_tables_tensor, block_tables_tensor->toDevice(DeviceType::kDeviceCUDA));
  ASSIGN_OR_THROW(output_tensor, output_tensor->toDevice(DeviceType::kDeviceCUDA));

  std::vector<const Tensor*> inputs = {q_tensor.get(),
                                       k_tensor.get(),
                                       v_tensor.get(),
                                       cu_seqlens_q_tensor.get(),
                                       cu_seqlens_kv_tensor.get(),
                                       block_tables_tensor.get()};
  std::vector<Tensor*> outputs = {output_tensor.get()};
  auto status = op.run(core::InferContext{}.setMaxSeqlenQ(max_seqlen_q), inputs, outputs);
  CHECK(status.ok()) << "GQAVarlenOp run failed: " << status.err();

  ASSIGN_OR_THROW(output_tensor, output_tensor->toDevice(DeviceType::kDeviceCPU));
  return output_tensor;
}

// ─────────────────────────────────────────────
//  test_store_kvcache_op_cuda
//
//  k / v:         [total_tokens, kv_heads, head_dim]
//  k_cache/v_cache: [total_slots, kv_heads, head_dim]  (pre-allocated pool)
//  slot_mapping:  [total_tokens]  int32
//
//  in-place op — results are written back into k_cache/v_cache (CPU numpy buffers) via copyFrom.
// ─────────────────────────────
void test_store_kvcache_op_cuda(TensorRef k_tensor,
                                TensorRef v_tensor,
                                TensorRef k_cache_tensor,
                                TensorRef v_cache_tensor,
                                TensorRef slot_mapping_tensor) {
  ::ginfer::core::op::StoreKVCacheOp op(DeviceType::kDeviceCUDA);

  ASSIGN_OR_THROW(k_tensor, k_tensor->toDevice(DeviceType::kDeviceCUDA));
  ASSIGN_OR_THROW(v_tensor, v_tensor->toDevice(DeviceType::kDeviceCUDA));
  ASSIGN_OR_THROW(slot_mapping_tensor, slot_mapping_tensor->toDevice(DeviceType::kDeviceCUDA));

  // k_cache / v_cache: create separate GPU copies so the original CPU tensors
  // (which wrap the numpy buffer) remain intact for copyFrom at the end.
  auto k_cache_gpu_res = Tensor::create(k_cache_tensor->dtype(), Shape(k_cache_tensor->shape()),
                                        DeviceType::kDeviceCUDA);
  CHECK(k_cache_gpu_res.ok()) << k_cache_gpu_res.err();
  auto k_cache_gpu = k_cache_gpu_res.value();

  auto v_cache_gpu_res = Tensor::create(v_cache_tensor->dtype(), Shape(v_cache_tensor->shape()),
                                        DeviceType::kDeviceCUDA);
  CHECK(v_cache_gpu_res.ok()) << v_cache_gpu_res.err();
  auto v_cache_gpu = v_cache_gpu_res.value();

  std::vector<const Tensor*> inputs = {k_tensor.get(), v_tensor.get(), k_cache_gpu.get(),
                                       v_cache_gpu.get(), slot_mapping_tensor.get()};
  std::vector<Tensor*> outputs = {};
  auto status = op.run(core::InferContext{}, inputs, outputs);
  CHECK(status.ok()) << "StoreKVCacheOp run failed: " << status.err();

  // Copy GPU results back into the original CPU tensors (numpy buffers)
  k_cache_tensor->copyFrom(k_cache_gpu);  // GPU → CPU
  v_cache_tensor->copyFrom(v_cache_gpu);  // GPU → CPU
}

REGISTER_PYBIND_TEST(test_gqa_varlen_op_cuda);
REGISTER_PYBIND_TEST(test_store_kvcache_op_cuda);

}  // namespace ginfer::test::pybind
