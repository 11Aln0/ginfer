#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <unordered_map>
#include "ginfer/op/layer.h"
#include "ginfer/test/pybind/type.h"

namespace py = pybind11;

namespace ginfer::test::pybind {

template <typename T>
py::array_t<T> test_add_layer_cuda(py::array_t<T> a_np, py::array_t<T> b_np) {
  using ginfer::common::DeviceType;
  using ginfer::memory::Buffer;
  using ginfer::tensor::DataType;
  using ginfer::tensor::Shape;
  using ginfer::tensor::Tensor;

  if (a_np.ndim() != b_np.ndim()) {
    throw std::runtime_error("Input arrays must have the same number of dimensions");
  }

  size_t ndim = a_np.ndim();
  std::vector<int64_t> shape_vec;
  size_t total = 1;

  for (size_t i = 0; i < ndim; i++) {
    if (a_np.shape(i) != b_np.shape(i)) {
      throw std::runtime_error("Input arrays must have the same shape");
    }
    int dim_size = static_cast<int>(a_np.shape(i));
    shape_vec.push_back(dim_size);
    total *= dim_size;
  }

  size_t bytes = total * sizeof(T);

  auto a_buf = std::make_shared<Buffer>(bytes, (void*)a_np.mutable_data(), DeviceType::kDeviceCPU);
  auto b_buf = std::make_shared<Buffer>(bytes, (void*)b_np.mutable_data(), DeviceType::kDeviceCPU);

  constexpr DataType dtype = tensor::DataTypeOf<T>::dtype;

  Tensor a_tensor(dtype, Shape(shape_vec), a_buf);
  Tensor b_tensor(dtype, Shape(shape_vec), b_buf);
  Tensor c_tensor(dtype, Shape(shape_vec), DeviceType::kDeviceCPU);

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
  py::array_t<T> c_np(shape_vec);
  std::memcpy(c_np.mutable_data(), c_tensor.data<T>(), bytes);
  return c_np;
}

template <typename T>
py::object dispatch_test(py::array a_np, py::array b_np) {
  return test_add_layer_cuda<T>(py::array_t<T>(a_np.request()), py::array_t<T>(b_np.request()));
}

py::object run_add_layer_cuda_test(py::array a_np, py::array b_np) {
  const std::string dtype_str = py::str(a_np.dtype());
  const std::unordered_map<std::string, std::function<py::object()>> dispatcher = {
      {"float32", [&]() { return dispatch_test<ginfer::type::Float32>(a_np, b_np); }},
      {"float16", [&]() { return dispatch_test<ginfer::type::Float16>(a_np, b_np); }},
      {"int32", [&]() { return dispatch_test<ginfer::type::Int32>(a_np, b_np); }},
      {"int8", [&]() { return dispatch_test<ginfer::type::Int8>(a_np, b_np); }},
  };
  return dispatcher.at(dtype_str)();
}

}  // namespace ginfer::test::pybind
