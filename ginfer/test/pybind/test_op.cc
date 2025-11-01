#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <unordered_map>
#include "ginfer/op/layer.h"

namespace py = pybind11;

namespace ginfer::test {

template <typename T, typename NPType = typename type::NumpyType<T>::type>
py::array_t<NPType> test_add_layer_cuda(py::array_t<NPType> a_np, py::array_t<NPType> b_np) {
  using ginfer::common::DeviceType;
  using ginfer::memory::Buffer;
  using ginfer::tensor::DataType;
  using ginfer::tensor::Shape;
  using ginfer::tensor::Tensor;

  // Ensure input shapes match
  if (a_np.shape(0) != b_np.shape(0) || a_np.shape(1) != b_np.shape(1)) {
    throw std::runtime_error("Input arrays must have the same shape");
  }

  int rows = static_cast<int>(a_np.shape(0));
  int cols = static_cast<int>(a_np.shape(1));
  size_t total = static_cast<size_t>(rows) * cols;
  size_t bytes = total * sizeof(NPType);

  auto a_buf = std::make_shared<Buffer>(bytes, (void*)a_np.mutable_data(), DeviceType::kDeviceCPU);
  auto b_buf = std::make_shared<Buffer>(bytes, (void*)b_np.mutable_data(), DeviceType::kDeviceCPU);

  constexpr DataType dtype = tensor::DataTypeOf<T>::dtype;

  Tensor a_tensor(dtype, Shape({rows, cols}), a_buf);
  Tensor b_tensor(dtype, Shape({rows, cols}), b_buf);
  Tensor c_tensor(dtype, Shape({rows, cols}), DeviceType::kDeviceCPU);

  ginfer::op::AddLayer add_layer(DeviceType::kDeviceCUDA, "add_layer_cuda");

  // Move tensors to GPU
  a_tensor.toDevice(DeviceType::kDeviceCUDA);
  b_tensor.toDevice(DeviceType::kDeviceCUDA);
  c_tensor.toDevice(DeviceType::kDeviceCUDA);

  // Run forward computation
  std::vector<const Tensor*> inputs = {&a_tensor, &b_tensor};
  auto status = add_layer.forward(inputs, &c_tensor);
  if (status.code() != ginfer::error::StatusCode::kSuccess) {
    throw std::runtime_error("AddLayer forward failed: " + status.msg());
  }

  // Copy result back to CPU
  c_tensor.toDevice(DeviceType::kDeviceCPU);

  // Convert result to NumPy array
  py::array_t<NPType> c_np({rows, cols});
  std::memcpy(c_np.mutable_data(), c_tensor.data<NPType>(), bytes);
  return c_np;
}

template <typename T>
py::object dispatch_test(py::array a_np, py::array b_np) {
  using NPType = typename type::NumpyType<T>::type;
  return test_add_layer_cuda<T, NPType>(a_np.cast<py::array_t<NPType>>(),
                                        b_np.cast<py::array_t<NPType>>());
}

py::object run_add_layer_cuda_test(py::array a_np, py::array b_np) {
  const std::string dtype_str = py::str(a_np.dtype());
  const std::unordered_map<std::string, std::function<py::object()>> dispatcher = {
      {"float32", [&]() { return dispatch_test<type::Float32>(a_np, b_np); }},
      {"float16", [&]() { return dispatch_test<type::Float16>(a_np, b_np); }},
      {"int32", [&]() { return dispatch_test<type::Int32>(a_np, b_np); }},
      {"int8", [&]() { return dispatch_test<type::Int8>(a_np, b_np); }},
  };
  return dispatcher.at(dtype_str)();
}

}  // namespace ginfer::test
