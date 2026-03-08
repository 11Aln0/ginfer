#include <glog/logging.h>
#include "ginfer/common/context.h"
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

TensorRef test_argmax_op_cuda(TensorRef input_tensor) {
  auto out_res = Tensor::create(DataType::kDataTypeInt64, Shape({1}), DeviceType::kDeviceCPU);
  CHECK(out_res.ok()) << out_res.err();
  auto output_tensor = out_res.value();

  ::ginfer::core::op::ArgmaxOp argmax_op(DeviceType::kDeviceCUDA);

  input_tensor->toDevice(DeviceType::kDeviceCUDA);
  output_tensor->toDevice(DeviceType::kDeviceCUDA);

  std::vector<const Tensor*> inputs = {input_tensor.get()};
  std::vector<Tensor*> outputs = {output_tensor.get()};
  auto status = argmax_op.run(common::InferContext{}, inputs, outputs);
  CHECK(status.ok()) << "ArgmaxOp run failed: " << status.err();

  output_tensor->toDevice(DeviceType::kDeviceCPU);
  return output_tensor;
}

REGISTER_PYBIND_TEST(test_argmax_op_cuda);

}  // namespace ginfer::test::pybind
