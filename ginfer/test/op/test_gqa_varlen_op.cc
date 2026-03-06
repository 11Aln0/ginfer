#include <glog/logging.h>
#include "ginfer/common/context.h"
#include "ginfer/op/op.h"
#include "ginfer/test/pybind/func_wrap.h"
#include "ginfer/test/pybind/test_registry.h"
#include "ginfer/test/pybind/type.h"

namespace py = pybind11;

namespace ginfer::test::pybind {

using common::DeviceType;
using tensor::DataType;
using tensor::Shape;
using tensor::Tensor;
using tensor::TensorRef;

// q: total_q_tokens, num_heads, head_dim]
// k, v: [num_blocks, block_size, kv_heads, head_dim]  (paged layout)
// cu_seqlens_q:  [batch + 1]  int32
// cu_seqlens_kv: [batch + 1]  int32
// block_tables:  [batch, max_blocks_per_seq] int32
TensorRef test_gqa_varlen_op_cuda(TensorRef q_tensor, TensorRef k_tensor, TensorRef v_tensor,
                                  TensorRef cu_seqlens_q_tensor, TensorRef cu_seqlens_kv_tensor,
                                  TensorRef block_tables_tensor, int max_seqlen_q, int paged_block_size) {
  auto out_res = Tensor::create(q_tensor->dtype(), Shape(q_tensor->shape()), DeviceType::kDeviceCPU);
  CHECK(out_res.ok()) << out_res.err();
  auto output_tensor = out_res.value();

  ::ginfer::op::GQAVarlenOp op(DeviceType::kDeviceCUDA, paged_block_size);

  // Move tensors to GPU
  q_tensor->toDevice(DeviceType::kDeviceCUDA);
  k_tensor->toDevice(DeviceType::kDeviceCUDA);
  v_tensor->toDevice(DeviceType::kDeviceCUDA);
  cu_seqlens_q_tensor->toDevice(DeviceType::kDeviceCUDA);
  cu_seqlens_kv_tensor->toDevice(DeviceType::kDeviceCUDA);
  block_tables_tensor->toDevice(DeviceType::kDeviceCUDA);
  output_tensor->toDevice(DeviceType::kDeviceCUDA);

  std::vector<const Tensor*> inputs = {
      q_tensor.get(),           k_tensor.get(), v_tensor.get(), cu_seqlens_q_tensor.get(), cu_seqlens_kv_tensor.get(),
      block_tables_tensor.get()};
  std::vector<Tensor*> outputs = {output_tensor.get()};
  auto status = op.run(common::InferContext{}.setMaxSeqlenQ(max_seqlen_q), inputs, outputs);
  CHECK(status.ok()) << "GQAVarlenOp run failed: " << status.err();

  output_tensor->toDevice(DeviceType::kDeviceCPU);
  return output_tensor;
}

REGISTER_PYBIND_TEST(test_gqa_varlen_op_cuda);

}  // namespace ginfer::test::pybind
