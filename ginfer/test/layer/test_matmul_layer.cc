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

Tensor test_matmul_layer_cuda(Tensor& a_tensor, Tensor& b_tensor) {
  DataType dtype = a_tensor.dtype();
  const Shape& a_shape = a_tensor.shape();
  const Shape& b_shape = b_tensor.shape();
  Tensor c_tensor(dtype, Shape({a_shape[0], b_shape[1]}), DeviceType::kDeviceCPU);

  ::ginfer::op::MatmulLayer matmul_layer(DeviceType::kDeviceCUDA, "matmul_layer_cuda");

  // Move tensors to GPU
  a_tensor.toDevice(DeviceType::kDeviceCUDA);
  b_tensor.toDevice(DeviceType::kDeviceCUDA);
  c_tensor.toDevice(DeviceType::kDeviceCUDA);

  // Run forward computation
  std::vector<const Tensor*> inputs = {&a_tensor, &b_tensor};
  auto status = matmul_layer.forward(inputs, &c_tensor);
  if (status.code() != ::ginfer::error::StatusCode::kSuccess) {
    throw std::runtime_error("MatmulLayer forward failed: " + status.msg());
  }

  // Copy result back to CPU
  c_tensor.toDevice(DeviceType::kDeviceCPU);

  return c_tensor;
}

REGISTER_PYBIND_TEST(test_matmul_layer_cuda);

}  // namespace ginfer::test::pybind