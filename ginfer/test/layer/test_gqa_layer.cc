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

Tensor test_gqa_layer_cuda(Tensor& q_tensor, Tensor& k_tensor, Tensor& v_tensor, int seq_len) {
  DataType dtype = q_tensor.dtype();
  const Shape& shape = q_tensor.shape();
  Tensor output_tensor(dtype, Shape(shape), DeviceType::kDeviceCPU);

  ::ginfer::op::GQALayer gqa_layer(DeviceType::kDeviceCUDA, "gqa_layer_cuda");
  gqa_layer.setSeqLen(seq_len);

  // Move tensors to GPU
  q_tensor.toDevice(DeviceType::kDeviceCUDA);
  k_tensor.toDevice(DeviceType::kDeviceCUDA);
  v_tensor.toDevice(DeviceType::kDeviceCUDA);
  output_tensor.toDevice(DeviceType::kDeviceCUDA);

  // Run forward computation
  std::vector<const Tensor*> inputs = {&q_tensor, &k_tensor, &v_tensor};
  auto status = gqa_layer.forward(inputs, &output_tensor);
  CHECK(status.code() == ::ginfer::error::StatusCode::kSuccess) << "GQALayer forward failed: " << status.msg();

  // Copy result back to CPU
  output_tensor.toDevice(DeviceType::kDeviceCPU);

  return output_tensor;
}

REGISTER_PYBIND_TEST(test_gqa_layer_cuda);

}  // namespace ginfer::test::pybind