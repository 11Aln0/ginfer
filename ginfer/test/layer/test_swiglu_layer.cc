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

Tensor test_swiglu_layer_cuda(Tensor& gate_tensor, Tensor& up_tensor) {
  DataType dtype = gate_tensor.dtype();
  const Shape& shape = gate_tensor.shape();
  Tensor output_tensor(dtype, Shape(shape), DeviceType::kDeviceCPU);

  ::ginfer::op::SwiGLULayer swiglu_layer(DeviceType::kDeviceCUDA, "swiglu_layer_cuda");

  gate_tensor.toDevice(DeviceType::kDeviceCUDA);
  up_tensor.toDevice(DeviceType::kDeviceCUDA);
  output_tensor.toDevice(DeviceType::kDeviceCUDA);

  std::vector<const Tensor*> inputs = {&gate_tensor, &up_tensor};
  auto status = swiglu_layer.forward(inputs, &output_tensor);
  CHECK(status.code() == ::ginfer::error::StatusCode::kSuccess) << "SwiGLULayer forward failed: " << status.msg();

  output_tensor.toDevice(DeviceType::kDeviceCPU);
  return output_tensor;
}

REGISTER_PYBIND_TEST(test_swiglu_layer_cuda);

}  // namespace ginfer::test::pybind