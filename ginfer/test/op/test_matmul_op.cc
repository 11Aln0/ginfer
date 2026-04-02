#include <cuda_runtime.h>
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

  ::ginfer::core::op::MatmulOp matmul_op(DeviceType::kDeviceCUDA);

  // Move tensors to GPU
  ASSIGN_OR_THROW(a_tensor, a_tensor->toDevice(DeviceType::kDeviceCUDA));
  ASSIGN_OR_THROW(b_tensor, b_tensor->toDevice(DeviceType::kDeviceCUDA));
  ASSIGN_OR_THROW(c_tensor, c_tensor->toDevice(DeviceType::kDeviceCUDA));

  // Run computation
  auto dev_ctx = ginfer::common::DeviceContext::create(DeviceType::kDeviceCUDA);
  std::vector<const Tensor*> inputs = {a_tensor.get(), b_tensor.get()};
  std::vector<Tensor*> outputs = {c_tensor.get()};
  auto status = matmul_op.run(core::InferContext{}.setDeviceContext(dev_ctx), inputs, outputs);
  CHECK(status.ok()) << "MatmulOp run failed: " << status.err();

  // Copy result back to CPU
  ASSIGN_OR_THROW(c_tensor, c_tensor->toDevice(DeviceType::kDeviceCPU));

  return c_tensor;
}

TensorRef test_matmul_op_with_bias_cuda(TensorRef a_tensor,
                                        TensorRef b_tensor,
                                        TensorRef bias_tensor) {
  const Shape& a_shape = a_tensor->shape();
  const Shape& b_shape = b_tensor->shape();
  auto shape = Shape({a_shape[0], b_shape[1]});
  if (a_shape.ndim() == 1) {
    shape = Shape({b_shape[1]});
  }
  auto c_res = Tensor::create(a_tensor->dtype(), shape, DeviceType::kDeviceCPU);
  CHECK(c_res.ok()) << c_res.err();
  auto c_tensor = c_res.value();

  ::ginfer::core::op::MatmulOp matmul_op(DeviceType::kDeviceCUDA);

  // Move tensors to GPU
  ASSIGN_OR_THROW(a_tensor, a_tensor->toDevice(DeviceType::kDeviceCUDA));
  ASSIGN_OR_THROW(b_tensor, b_tensor->toDevice(DeviceType::kDeviceCUDA));
  ASSIGN_OR_THROW(bias_tensor, bias_tensor->toDevice(DeviceType::kDeviceCUDA));
  ASSIGN_OR_THROW(c_tensor, c_tensor->toDevice(DeviceType::kDeviceCUDA));

  // Run computation
  auto dev_ctx = ginfer::common::DeviceContext::create(DeviceType::kDeviceCUDA);
  std::vector<const Tensor*> inputs = {a_tensor.get(), b_tensor.get(), bias_tensor.get()};
  std::vector<Tensor*> outputs = {c_tensor.get()};
  auto status = matmul_op.run(core::InferContext{}.setDeviceContext(dev_ctx), inputs, outputs);
  CHECK(status.ok()) << "MatmulOp run failed: " << status.err();

  // Copy result back to CPU
  ASSIGN_OR_THROW(c_tensor, c_tensor->toDevice(DeviceType::kDeviceCPU));

  return c_tensor;
}

REGISTER_PYBIND_TEST(test_matmul_op_cuda);
REGISTER_PYBIND_TEST(test_matmul_op_with_bias_cuda);

}  // namespace ginfer::test::pybind
