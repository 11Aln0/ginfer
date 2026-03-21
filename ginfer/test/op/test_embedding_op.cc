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

TensorRef test_embedding_op_cuda(TensorRef input_tensor, TensorRef weight_tensor) {
  const Shape& input_shape = input_tensor->shape();
  const Shape& weight_shape = weight_tensor->shape();
  int64_t embedding_dim = weight_shape[weight_shape.ndim() - 1];

  // Output shape: input_shape + [embedding_dim]
  std::vector<int64_t> out_dims;
  for (size_t i = 0; i < input_shape.ndim(); i++) {
    out_dims.push_back(input_shape[i]);
  }
  out_dims.push_back(embedding_dim);
  auto out_res = Tensor::create(weight_tensor->dtype(), Shape(out_dims), DeviceType::kDeviceCPU);
  CHECK(out_res.ok()) << out_res.err();
  auto output_tensor = out_res.value();

  ::ginfer::core::op::EmbeddingOp embedding_op(DeviceType::kDeviceCUDA);

  ASSIGN_OR_THROW(input_tensor, input_tensor->toDevice(DeviceType::kDeviceCUDA));
  ASSIGN_OR_THROW(weight_tensor, weight_tensor->toDevice(DeviceType::kDeviceCUDA));
  ASSIGN_OR_THROW(output_tensor, output_tensor->toDevice(DeviceType::kDeviceCUDA));

  std::vector<const Tensor*> inputs = {input_tensor.get(), weight_tensor.get()};
  std::vector<Tensor*> outputs = {output_tensor.get()};
  auto status = embedding_op.run(core::InferContext{}, inputs, outputs);
  CHECK(status.ok()) << "EmbeddingOp run failed: " << status.err();

  ASSIGN_OR_THROW(output_tensor, output_tensor->toDevice(DeviceType::kDeviceCPU));
  return output_tensor;
}

REGISTER_PYBIND_TEST(test_embedding_op_cuda);

}  // namespace ginfer::test::pybind
