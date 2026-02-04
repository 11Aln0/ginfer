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

Tensor test_rope_layer_cuda(Tensor& input_tensor, int head_dim, int start_pos, int end_pos, float rope_theta) {
  DataType dtype = input_tensor.dtype();
  const Shape& shape = input_tensor.shape();
  Tensor output_tensor(dtype, Shape(shape), DeviceType::kDeviceCPU);

  int max_seq_len = end_pos;
  ::ginfer::op::ROPELayer rope_layer(DeviceType::kDeviceCUDA, "rope_layer_cuda", head_dim, max_seq_len, rope_theta);
  rope_layer.updateCache(start_pos, end_pos);

  input_tensor.toDevice(DeviceType::kDeviceCUDA);
  output_tensor.toDevice(DeviceType::kDeviceCUDA);

  std::vector<const Tensor*> inputs = {&input_tensor};
  auto status = rope_layer.forward(inputs, &output_tensor);
  CHECK(status.code() == ::ginfer::error::StatusCode::kSuccess) << "ROPELayer forward failed: " << status.msg();

  output_tensor.toDevice(DeviceType::kDeviceCPU);
  return output_tensor;
}

REGISTER_PYBIND_TEST(test_rope_layer_cuda);

}  // namespace ginfer::test::pybind