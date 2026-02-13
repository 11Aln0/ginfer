#include <glog/logging.h>
#include "ginfer/op/op.h"
#include "ginfer/test/pybind/func_wrap.h"
#include "ginfer/test/pybind/test_registry.h"
#include "ginfer/test/pybind/type.h"

namespace py = pybind11;

namespace ginfer::test::pybind {

using common::DeviceType;
using memory::Buffer;
using tensor::DataType;
using tensor::Shape;
using tensor::Tensor;

Tensor test_rope_op_cuda(Tensor& input_tensor, int head_dim, int start_pos, int end_pos, float rope_theta) {
  DataType dtype = input_tensor.dtype();
  const Shape& shape = input_tensor.shape();
  Tensor output_tensor(dtype, Shape(shape), DeviceType::kDeviceCPU);

  int max_seq_len = end_pos;

  // Create sin/cos cache tensors
  Tensor sin_cache(DataType::kDataTypeFloat32, Shape({max_seq_len, head_dim / 2}), DeviceType::kDeviceCUDA);
  Tensor cos_cache(DataType::kDataTypeFloat32, Shape({max_seq_len, head_dim / 2}), DeviceType::kDeviceCUDA);

  // Compute sin/cos cache
  ::ginfer::op::RotaryEmbeddingOp rotary_embed_op(DeviceType::kDeviceCUDA, rope_theta);
  auto pos_ids = Tensor(DataType::kDataTypeInt64, Shape({2}), DeviceType::kDeviceCPU);
  pos_ids.data<int64_t>()[0] = start_pos;
  pos_ids.data<int64_t>()[1] = end_pos;
  std::vector<const Tensor*> embed_inputs = {&pos_ids};
  std::vector<Tensor*> embed_outputs = {&sin_cache, &cos_cache};
  auto embed_status = rotary_embed_op.run(embed_inputs, embed_outputs);
  CHECK(embed_status.code() == ::ginfer::error::StatusCode::kSuccess)
      << "RotaryEmbeddingOp run failed: " << embed_status.msg();

  // Run ROPE
  ::ginfer::op::ROPEOp rope_op(DeviceType::kDeviceCUDA);

  input_tensor.toDevice(DeviceType::kDeviceCUDA);
  output_tensor.toDevice(DeviceType::kDeviceCUDA);

  std::vector<const Tensor*> inputs = {&input_tensor, &sin_cache, &cos_cache};
  std::vector<Tensor*> outputs = {&output_tensor};
  auto status = rope_op.run(inputs, outputs);
  CHECK(status.code() == ::ginfer::error::StatusCode::kSuccess) << "ROPEOp run failed: " << status.msg();

  output_tensor.toDevice(DeviceType::kDeviceCPU);
  return output_tensor;
}

REGISTER_PYBIND_TEST(test_rope_op_cuda);

}  // namespace ginfer::test::pybind
