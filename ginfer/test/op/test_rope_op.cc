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
  Tensor sin_cache(DataType::kDataTypeFloat32, Shape({max_seq_len, head_dim / 2}), DeviceType::kDeviceCPU);
  Tensor cos_cache(DataType::kDataTypeFloat32, Shape({max_seq_len, head_dim / 2}), DeviceType::kDeviceCPU);

  sin_cache.toDevice(DeviceType::kDeviceCUDA);
  cos_cache.toDevice(DeviceType::kDeviceCUDA);

  // Compute sin/cos cache
  ::ginfer::op::ROPESinCosCacheOp cache_op(DeviceType::kDeviceCUDA, head_dim, max_seq_len, rope_theta);
  std::vector<const Tensor*> cache_inputs = {};
  std::vector<Tensor*> cache_outputs = {&sin_cache, &cos_cache};
  auto cache_status = cache_op.run(cache_inputs, cache_outputs);
  CHECK(cache_status.code() == ::ginfer::error::StatusCode::kSuccess)
      << "ROPESinCosCacheOp run failed: " << cache_status.msg();

  // Run ROPE
  ::ginfer::op::ROPEOp rope_op(DeviceType::kDeviceCUDA, head_dim, max_seq_len, rope_theta);

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
