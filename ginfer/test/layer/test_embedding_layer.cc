#include <glog/logging.h>
#include "ginfer/op/layer.h"
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

Tensor test_embedding_layer_cuda(Tensor& input_tensor, Tensor& weight_tensor) {
  DataType dtype = weight_tensor.dtype();
  const Shape& input_shape = input_tensor.shape();
  const Shape& weight_shape = weight_tensor.shape();
  int64_t embedding_dim = weight_shape[weight_shape.ndim() - 1];

  // Output shape: input_shape + [embedding_dim]
  std::vector<int64_t> out_dims;
  for (size_t i = 0; i < input_shape.ndim(); i++) {
    out_dims.push_back(input_shape[i]);
  }
  out_dims.push_back(embedding_dim);
  Tensor output_tensor(dtype, Shape(out_dims), DeviceType::kDeviceCPU);

  ::ginfer::op::EmbeddingLayer emb_layer(DeviceType::kDeviceCUDA, "embedding_layer_cuda");
  emb_layer.setWeight(0, std::make_shared<Tensor>(weight_tensor));
  emb_layer.toDevice(DeviceType::kDeviceCUDA);

  input_tensor.toDevice(DeviceType::kDeviceCUDA);
  output_tensor.toDevice(DeviceType::kDeviceCUDA);

  std::vector<const Tensor*> inputs = {&input_tensor};
  auto status = emb_layer.forward(inputs, &output_tensor);
  CHECK(status.code() == ::ginfer::error::StatusCode::kSuccess) << "EmbeddingLayer forward failed: " << status.msg();

  output_tensor.toDevice(DeviceType::kDeviceCPU);
  return output_tensor;
}

REGISTER_PYBIND_TEST(test_embedding_layer_cuda);

}  // namespace ginfer::test::pybind
