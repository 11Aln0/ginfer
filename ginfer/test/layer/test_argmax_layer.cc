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

Tensor test_argmax_layer_cuda(Tensor& input_tensor) {
  Tensor output_tensor(DataType::kDataTypeInt64, Shape({1}), DeviceType::kDeviceCPU);

  ::ginfer::op::ArgmaxLayer argmax_layer(DeviceType::kDeviceCUDA, "argmax_layer_cuda");

  input_tensor.toDevice(DeviceType::kDeviceCUDA);
  output_tensor.toDevice(DeviceType::kDeviceCUDA);

  std::vector<const Tensor*> inputs = {&input_tensor};
  auto status = argmax_layer.forward(inputs, &output_tensor);
  CHECK(status.code() == ::ginfer::error::StatusCode::kSuccess) << "ArgmaxLayer forward failed: " << status.msg();

  output_tensor.toDevice(DeviceType::kDeviceCPU);
  return output_tensor;
}

REGISTER_PYBIND_TEST(test_argmax_layer_cuda);

}  // namespace ginfer::test::pybind