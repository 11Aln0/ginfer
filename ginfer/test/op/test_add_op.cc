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

TensorRef test_add_op_cuda(TensorRef a_tensor, TensorRef b_tensor) {
  auto c_res = Tensor::create(a_tensor->dtype(), Shape(a_tensor->shape()), DeviceType::kDeviceCPU);
  CHECK(c_res.ok()) << c_res.err();
  auto c_tensor = c_res.value();

  ::ginfer::core::op::AddOp add_op(DeviceType::kDeviceCUDA);

  // Move tensors to GPU
  ASSIGN_OR_THROW(a_tensor, a_tensor->toDevice(DeviceType::kDeviceCUDA));
  ASSIGN_OR_THROW(b_tensor, b_tensor->toDevice(DeviceType::kDeviceCUDA));
  ASSIGN_OR_THROW(c_tensor, c_tensor->toDevice(DeviceType::kDeviceCUDA));

  // Run computation
  std::vector<const Tensor*> inputs = {a_tensor.get(), b_tensor.get()};
  std::vector<Tensor*> outputs = {c_tensor.get()};
  auto status = add_op.run(core::InferContext{}, inputs, outputs);
  CHECK(status.ok()) << "AddOp run failed: " << status.err();

  // Copy result back to CPU
  ASSIGN_OR_THROW(c_tensor, c_tensor->toDevice(DeviceType::kDeviceCPU));

  return c_tensor;
}

REGISTER_PYBIND_TEST(test_add_op_cuda);

}  // namespace ginfer::test::pybind
