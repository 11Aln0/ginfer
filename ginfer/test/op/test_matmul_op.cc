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

Tensor test_matmul_op_cuda(Tensor& a_tensor, Tensor& b_tensor) {
  DataType dtype = a_tensor.dtype();
  const Shape& a_shape = a_tensor.shape();
  const Shape& b_shape = b_tensor.shape();
  auto shape = Shape({a_shape[0], b_shape[1]});
  if (a_shape.ndim() == 1) {
    shape = Shape({b_shape[1]});
  }
  Tensor c_tensor(dtype, shape, DeviceType::kDeviceCPU);

  ::ginfer::op::MatmulOp matmul_op(DeviceType::kDeviceCUDA);

  // Move tensors to GPU
  a_tensor.toDevice(DeviceType::kDeviceCUDA);
  b_tensor.toDevice(DeviceType::kDeviceCUDA);
  c_tensor.toDevice(DeviceType::kDeviceCUDA);

  // Run computation
  std::vector<const Tensor*> inputs = {&a_tensor, &b_tensor};
  std::vector<Tensor*> outputs = {&c_tensor};
  auto status = matmul_op.run(inputs, outputs);
  CHECK(status.code() == ::ginfer::error::StatusCode::kSuccess) << "MatmulOp run failed: " << status.msg();

  // Copy result back to CPU
  c_tensor.toDevice(DeviceType::kDeviceCPU);

  return c_tensor;
}

REGISTER_PYBIND_TEST(test_matmul_op_cuda);

}  // namespace ginfer::test::pybind
