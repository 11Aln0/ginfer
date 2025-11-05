#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <execution>
#include <numeric>
#include <unordered_map>
#include "ginfer/op/layer.h"
#include "ginfer/test/pybind/type.h"

namespace py = pybind11;

namespace ginfer::test::pybind {
template <typename T>
py::array_t<T> test_rmsnorm_layer_cuda(py::array_t<T> input_np, py::array_t<T> gamma_np, float epsilon) {
  using ginfer::common::DeviceType;
  using ginfer::memory::Buffer;
  using ginfer::tensor::DataType;
  using ginfer::tensor::Shape;
  using ginfer::tensor::Tensor;

  size_t ndim = input_np.ndim();
  std::vector<int64_t> shape;
  std::vector<int64_t> gamma_shape;

  for (size_t i = 0; i < ndim; i++) {
    auto dim_size = input_np.shape(i);
    shape.push_back(dim_size);
  }

  for (size_t i = 0; i < gamma_np.ndim(); i++) {
    auto dim_size = gamma_np.shape(i);
    gamma_shape.push_back(dim_size);
  }

  int64_t bytes = std::accumulate(shape.begin(), shape.end(), 1LL, std::multiplies<int64_t>()) * sizeof(T);
  int64_t gamma_bytes = std::accumulate(gamma_shape.begin(), gamma_shape.end(), 1LL, std::multiplies<int64_t>()) * sizeof(T);

  auto input_buf = std::make_shared<Buffer>(bytes, (void*)input_np.mutable_data(), DeviceType::kDeviceCPU);
  auto gamma_buf = std::make_shared<Buffer>(gamma_bytes, (void*)gamma_np.mutable_data(), DeviceType::kDeviceCPU);

  constexpr DataType dtype = tensor::DataTypeOf<T>::dtype;

  Tensor input_tensor(dtype, Shape(shape), input_buf);
  Tensor gamma_tensor(dtype, Shape(gamma_shape), gamma_buf);
  Tensor output_tensor(dtype, Shape(shape), DeviceType::kDeviceCPU);

  ginfer::op::RMSNormLayer rmsnorm_layer(DeviceType::kDeviceCUDA, "rmsnorm_layer_cuda", epsilon);
  rmsnorm_layer.setWeight(0, std::make_shared<Tensor>(gamma_tensor));
  rmsnorm_layer.toDevice(DeviceType::kDeviceCUDA);

  // Move tensors to GPU
  input_tensor.toDevice(DeviceType::kDeviceCUDA);
  output_tensor.toDevice(DeviceType::kDeviceCUDA);

  // Run forward computation
  std::vector<const Tensor*> inputs = {&input_tensor};
  auto status = rmsnorm_layer.forward(inputs, &output_tensor);
  if (status.code() != ginfer::error::StatusCode::kSuccess) {
    throw std::runtime_error("RMSNormLayer forward failed: " + status.msg());
  }

  // Copy result back to CPU
  output_tensor.toDevice(DeviceType::kDeviceCPU);

  // Convert result to NumPy array
  py::array_t<T> output_np(shape);
  std::memcpy(output_np.mutable_data(), output_tensor.data<T>(), bytes);
  return output_np;
}

template <typename T>
py::object dispatch_rmsnorm_layer_test(py::array input_np, py::array gamma_np, float epsilon) {
  return test_rmsnorm_layer_cuda<T>(py::array_t<T>(input_np.request()), py::array_t<T>(gamma_np.request()), epsilon);
}

py::object run_rmsnorm_layer_cuda_test(py::array input_np, py::array gamma_np, float epsilon) {
  const std::string dtype_str = py::str(input_np.dtype());
  const std::unordered_map<std::string, std::function<py::object()>> dispatcher = {
      {"float32", [&]() { return dispatch_rmsnorm_layer_test<ginfer::type::Float32>(input_np, gamma_np, epsilon); }},
      {"float16", [&]() { return dispatch_rmsnorm_layer_test<ginfer::type::Float16>(input_np, gamma_np, epsilon); }},
      {"int32", [&]() { return dispatch_rmsnorm_layer_test<ginfer::type::Int32>(input_np, gamma_np, epsilon); }},
      {"int8", [&]() { return dispatch_rmsnorm_layer_test<ginfer::type::Int8>(input_np, gamma_np, epsilon); }},
  };
  return dispatcher.at(dtype_str)();
}

}  // namespace ginfer::test::pybind