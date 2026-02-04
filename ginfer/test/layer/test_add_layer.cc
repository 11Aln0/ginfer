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

Tensor test_add_layer_cuda(Tensor& a_tensor, Tensor& b_tensor) {
  DataType dtype = a_tensor.dtype();
  const Shape& shape = a_tensor.shape();
  Tensor c_tensor(dtype, Shape(shape), DeviceType::kDeviceCPU);

  ::ginfer::op::AddLayer add_layer(DeviceType::kDeviceCUDA, "add_layer_cuda");

  // Move tensors to GPU
  a_tensor.toDevice(DeviceType::kDeviceCUDA);
  b_tensor.toDevice(DeviceType::kDeviceCUDA);
  c_tensor.toDevice(DeviceType::kDeviceCUDA);

  // Run forward computation
  std::vector<const Tensor*> inputs = {&a_tensor, &b_tensor};
  auto status = add_layer.forward(inputs, &c_tensor);
  CHECK(status.code() == ::ginfer::error::StatusCode::kSuccess) << "AddLayer forward failed: " << status.msg();

  // Copy result back to CPU
  c_tensor.toDevice(DeviceType::kDeviceCPU);

  return c_tensor;
}

REGISTER_PYBIND_TEST(test_add_layer_cuda);

}  // namespace ginfer::test::pybind
