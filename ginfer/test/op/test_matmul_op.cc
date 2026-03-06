#include <glog/logging.h>
#include "ginfer/common/context.h"
#include "ginfer/op/op.h"
#include "ginfer/test/pybind/func_wrap.h"
#include "ginfer/test/pybind/test_registry.h"
#include "ginfer/test/pybind/type.h"

namespace py = pybind11;

namespace ginfer::test::pybind {

using common::DeviceType;
using tensor::DataType;
using tensor::Shape;
using tensor::Tensor;
using tensor::TensorRef;

TensorRef test_matmul_op_cuda(TensorRef a_tensor, TensorRef b_tensor) {
  const Shape& a_shape = a_tensor->shape();
  const Shape& b_shape = b_tensor->shape();
  auto shape = Shape({a_shape[0], b_shape[1]});
  if (a_shape.ndim() == 1) {
    shape = Shape({b_shape[1]});
  }
  auto c_res = Tensor::create(a_tensor->dtype(), shape, DeviceType::kDeviceCPU);
  CHECK(c_res.ok()) << c_res.err();
  auto c_tensor = c_res.value();

  ::ginfer::op::MatmulOp matmul_op(DeviceType::kDeviceCUDA);

  // Move tensors to GPU
  a_tensor->toDevice(DeviceType::kDeviceCUDA);
  b_tensor->toDevice(DeviceType::kDeviceCUDA);
  c_tensor->toDevice(DeviceType::kDeviceCUDA);

  // Run computation
  std::vector<const Tensor*> inputs = {a_tensor.get(), b_tensor.get()};
  std::vector<Tensor*> outputs = {c_tensor.get()};
  auto status = matmul_op.run(common::InferContext{}, inputs, outputs);
  CHECK(status.ok()) << "MatmulOp run failed: " << status.err();

  // Copy result back to CPU
  c_tensor->toDevice(DeviceType::kDeviceCPU);

  return c_tensor;
}

TensorRef test_matmul_op_with_bias_cuda(TensorRef a_tensor, TensorRef b_tensor, TensorRef bias_tensor) {
  const Shape& a_shape = a_tensor->shape();
  const Shape& b_shape = b_tensor->shape();
  auto shape = Shape({a_shape[0], b_shape[1]});
  if (a_shape.ndim() == 1) {
    shape = Shape({b_shape[1]});
  }
  auto c_res = Tensor::create(a_tensor->dtype(), shape, DeviceType::kDeviceCPU);
  CHECK(c_res.ok()) << c_res.err();
  auto c_tensor = c_res.value();

  ::ginfer::op::MatmulOp matmul_op(DeviceType::kDeviceCUDA);

  // Move tensors to GPU
  a_tensor->toDevice(DeviceType::kDeviceCUDA);
  b_tensor->toDevice(DeviceType::kDeviceCUDA);
  bias_tensor->toDevice(DeviceType::kDeviceCUDA);
  c_tensor->toDevice(DeviceType::kDeviceCUDA);

  // Run computation
  std::vector<const Tensor*> inputs = {a_tensor.get(), b_tensor.get(), bias_tensor.get()};
  std::vector<Tensor*> outputs = {c_tensor.get()};
  auto status = matmul_op.run(common::InferContext{}, inputs, outputs);
  CHECK(status.ok()) << "MatmulOp run failed: " << status.err();

  // Copy result back to CPU
  c_tensor->toDevice(DeviceType::kDeviceCPU);

  return c_tensor;
}

REGISTER_PYBIND_TEST(test_matmul_op_cuda);
REGISTER_PYBIND_TEST(test_matmul_op_with_bias_cuda);

}  // namespace ginfer::test::pybind
