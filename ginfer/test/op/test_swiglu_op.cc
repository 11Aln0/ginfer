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

TensorRef test_swiglu_op_cuda(TensorRef gate_tensor, TensorRef up_tensor) {
  auto out_res =
      Tensor::create(gate_tensor->dtype(), Shape(gate_tensor->shape()), DeviceType::kDeviceCPU);
  CHECK(out_res.ok()) << out_res.err();
  auto output_tensor = out_res.value();

  ::ginfer::core::op::SwiGLUOp swiglu_op(DeviceType::kDeviceCUDA);

  ASSIGN_OR_THROW(gate_tensor, gate_tensor->toDevice(DeviceType::kDeviceCUDA));
  ASSIGN_OR_THROW(up_tensor, up_tensor->toDevice(DeviceType::kDeviceCUDA));
  ASSIGN_OR_THROW(output_tensor, output_tensor->toDevice(DeviceType::kDeviceCUDA));

  std::vector<const Tensor*> inputs = {gate_tensor.get(), up_tensor.get()};
  std::vector<Tensor*> outputs = {output_tensor.get()};
  auto status = swiglu_op.run(core::InferContext{}, inputs, outputs);
  CHECK(status.ok()) << "SwiGLUOp run failed: " << status.err();

  ASSIGN_OR_THROW(output_tensor, output_tensor->toDevice(DeviceType::kDeviceCPU));
  return output_tensor;
}

REGISTER_PYBIND_TEST(test_swiglu_op_cuda);

}  // namespace ginfer::test::pybind
