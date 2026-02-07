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

Tensor test_swiglu_op_cuda(Tensor& gate_tensor, Tensor& up_tensor) {
  DataType dtype = gate_tensor.dtype();
  const Shape& shape = gate_tensor.shape();
  Tensor output_tensor(dtype, Shape(shape), DeviceType::kDeviceCPU);

  ::ginfer::op::SwiGLUOp swiglu_op(DeviceType::kDeviceCUDA);

  gate_tensor.toDevice(DeviceType::kDeviceCUDA);
  up_tensor.toDevice(DeviceType::kDeviceCUDA);
  output_tensor.toDevice(DeviceType::kDeviceCUDA);

  std::vector<const Tensor*> inputs = {&gate_tensor, &up_tensor};
  std::vector<Tensor*> outputs = {&output_tensor};
  auto status = swiglu_op.run(inputs, outputs);
  CHECK(status.code() == ::ginfer::error::StatusCode::kSuccess) << "SwiGLUOp run failed: " << status.msg();

  output_tensor.toDevice(DeviceType::kDeviceCPU);
  return output_tensor;
}

REGISTER_PYBIND_TEST(test_swiglu_op_cuda);

}  // namespace ginfer::test::pybind
