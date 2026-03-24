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

TensorRef test_gqa_op_cuda(TensorRef q_tensor, TensorRef k_tensor, TensorRef v_tensor) {
  auto out_res =
      Tensor::create(q_tensor->dtype(), Shape(q_tensor->shape()), DeviceType::kDeviceCPU);
  CHECK(out_res.ok()) << out_res.err();
  auto output_tensor = out_res.value();

  ::ginfer::core::op::GQAOp gqa_op(DeviceType::kDeviceCUDA);

  // Move tensors to GPU
  ASSIGN_OR_THROW(q_tensor, q_tensor->toDevice(DeviceType::kDeviceCUDA));
  ASSIGN_OR_THROW(k_tensor, k_tensor->toDevice(DeviceType::kDeviceCUDA));
  ASSIGN_OR_THROW(v_tensor, v_tensor->toDevice(DeviceType::kDeviceCUDA));
  ASSIGN_OR_THROW(output_tensor, output_tensor->toDevice(DeviceType::kDeviceCUDA));

  // Run computation
  auto dev_ctx = ginfer::common::DeviceContext::create(DeviceType::kDeviceCUDA);
  std::vector<const Tensor*> inputs = {q_tensor.get(), k_tensor.get(), v_tensor.get()};
  std::vector<Tensor*> outputs = {output_tensor.get()};
  auto status = gqa_op.run(core::InferContext{}.setDeviceContext(dev_ctx), inputs, outputs);
  CHECK(status.ok()) << "GQAOp run failed: " << status.err();

  // Copy result back to CPU
  ASSIGN_OR_THROW(output_tensor, output_tensor->toDevice(DeviceType::kDeviceCPU));

  return output_tensor;
}

REGISTER_PYBIND_TEST(test_gqa_op_cuda);

}  // namespace ginfer::test::pybind
