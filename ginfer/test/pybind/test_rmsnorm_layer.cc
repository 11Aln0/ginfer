#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <execution>
#include <numeric>
#include <unordered_map>
#include "ginfer/op/layer.h"
#include "ginfer/test/pybind/type.h"

namespace py = pybind11;

namespace ginfer::test::pybind {
using common::DeviceType;
using memory::Buffer;
using tensor::DataType;
using tensor::Shape;
using tensor::Tensor;

Tensor test_rmsnorm_layer_cuda(Tensor& input_tensor, Tensor& gamma_tensor, float epsilon) {
  DataType dtype = input_tensor.dtype();
  const Shape& shape = input_tensor.shape();
  Tensor output_tensor(dtype, Shape(shape), DeviceType::kDeviceCPU);

  ::ginfer::op::RMSNormLayer rmsnorm_layer(DeviceType::kDeviceCUDA, "rmsnorm_layer_cuda", epsilon);
  rmsnorm_layer.setWeight(0, std::make_shared<Tensor>(gamma_tensor));
  rmsnorm_layer.toDevice(DeviceType::kDeviceCUDA);

  // Move tensors to GPU
  input_tensor.toDevice(DeviceType::kDeviceCUDA);
  output_tensor.toDevice(DeviceType::kDeviceCUDA);

  // Run forward computation
  std::vector<const Tensor*> inputs = {&input_tensor};
  auto status = rmsnorm_layer.forward(inputs, &output_tensor);
  if (status.code() != ::ginfer::error::StatusCode::kSuccess) {
    throw std::runtime_error("RMSNormLayer forward failed: " + status.msg());
  }

  // Copy result back to CPU
  output_tensor.toDevice(DeviceType::kDeviceCPU);

  return output_tensor;
}

}  // namespace ginfer::test::pybind