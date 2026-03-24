#include <glog/logging.h>
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

TensorRef test_rope_op_cuda(
    TensorRef input_tensor, int head_dim, int start_pos, int end_pos, float rope_theta) {
  auto out_res =
      Tensor::create(input_tensor->dtype(), Shape(input_tensor->shape()), DeviceType::kDeviceCPU);
  CHECK(out_res.ok()) << out_res.err();
  auto output_tensor = out_res.value();

  int max_seq_len = end_pos - start_pos + 1;

  // Create sin/cos cache tensors
  auto sin_res = Tensor::create(DataType::kDataTypeFloat32, Shape({max_seq_len, head_dim / 2}),
                                DeviceType::kDeviceCUDA);
  CHECK(sin_res.ok()) << sin_res.err();
  auto sin_cache = sin_res.value();

  auto cos_res = Tensor::create(DataType::kDataTypeFloat32, Shape({max_seq_len, head_dim / 2}),
                                DeviceType::kDeviceCUDA);
  CHECK(cos_res.ok()) << cos_res.err();
  auto cos_cache = cos_res.value();

  // Compute sin/cos cache
  ::ginfer::core::op::RotaryEmbeddingOp rotary_embed_op(DeviceType::kDeviceCUDA, rope_theta);
  auto pos_res = Tensor::create(DataType::kDataTypeInt64, Shape({2}), DeviceType::kDeviceCPU);
  CHECK(pos_res.ok()) << pos_res.err();
  auto pos_ids = pos_res.value();
  pos_ids->data<int64_t>()[0] = start_pos;
  pos_ids->data<int64_t>()[1] = end_pos;
  auto dev_ctx = ginfer::common::DeviceContext::create(DeviceType::kDeviceCUDA);
  std::vector<const Tensor*> embed_inputs = {pos_ids.get()};
  std::vector<Tensor*> embed_outputs = {sin_cache.get(), cos_cache.get()};
  auto embed_status =
      rotary_embed_op.run(core::InferContext{}.setDeviceContext(dev_ctx), embed_inputs, embed_outputs);
  CHECK(embed_status.ok()) << "RotaryEmbeddingOp run failed: " << embed_status.err();

  // Run ROPE
  ::ginfer::core::op::ROPEOp rope_op(DeviceType::kDeviceCUDA);

  int seq_len = input_tensor->shape()[0];
  auto positions_res =
      Tensor::create(DataType::kDataTypeInt32, Shape({seq_len}), DeviceType::kDeviceCPU);
  CHECK(positions_res.ok()) << positions_res.err();
  auto positions = positions_res.value();
  auto positions_data = positions->data<int32_t>();
  for (int i = 0; i < seq_len; ++i) {
    positions_data[i] = start_pos + i;
  }

  ASSIGN_OR_THROW(input_tensor, input_tensor->toDevice(DeviceType::kDeviceCUDA));
  ASSIGN_OR_THROW(positions, positions->toDevice(DeviceType::kDeviceCUDA));
  ASSIGN_OR_THROW(output_tensor, output_tensor->toDevice(DeviceType::kDeviceCUDA));

  std::vector<const Tensor*> inputs = {input_tensor.get(), positions.get(), sin_cache.get(),
                                       cos_cache.get()};
  std::vector<Tensor*> outputs = {output_tensor.get()};
  auto status = rope_op.run(core::InferContext{}.setDeviceContext(dev_ctx), inputs, outputs);
  CHECK(status.ok()) << "ROPEOp run failed: " << status.err();

  ASSIGN_OR_THROW(output_tensor, output_tensor->toDevice(DeviceType::kDeviceCPU));
  return output_tensor;
}

REGISTER_PYBIND_TEST(test_rope_op_cuda);

}  // namespace ginfer::test::pybind
