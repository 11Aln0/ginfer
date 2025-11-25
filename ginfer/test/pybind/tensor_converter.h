#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <array>
#include "ginfer/tensor/tensor.h"
#include "ginfer/test/pybind/type.h"

namespace ginfer::test::pybind {

namespace py = pybind11;

using common::DeviceType;
using memory::Buffer;
using tensor::DataType;
using tensor::dTypeSize;
using tensor::Layout;
using tensor::Shape;
using tensor::Tensor;

class NumpyToTensorConverter {
 public:
  static Tensor convert(py::array arr) {
    size_t ndim = arr.ndim();
    std::vector<int64_t> shape;
    for (int i = 0; i < ndim; ++i) {
      shape.push_back(arr.shape(i));
    }

    DataType dtype = type::numpyDtypeToTensorDtype(arr.dtype());
    int64_t bytes = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int64_t>()) * dTypeSize(dtype);
    auto buf = std::make_shared<Buffer>(bytes, (void*)arr.mutable_data(), DeviceType::kDeviceCPU);

    Layout layout = Layout::kLayoutRowMajor;
    if ((bool)(arr.flags() & py::array::f_style)) {
      layout = Layout::kLayoutColMajor;
    }

    return tensor::Tensor(dtype, Shape(shape), buf, layout);
  }

  static py::array convert_back(const Tensor& tensor) {
    const Shape& shape = tensor.shape();
    std::vector<size_t> np_shape;
    for (size_t i = 0; i < shape.ndim(); ++i) {
      np_shape.push_back(shape[i]);
    }

    py::dtype np_dtype = type::tensorDtypeToNumpyDtype(tensor.dtype());
    py::array np_array(np_dtype, np_shape);
    size_t nbytes = tensor.nbytes();
    std::memcpy(np_array.mutable_data(), tensor.data<void>(), nbytes);
    return np_array;
  }
};

}  // namespace ginfer::test::pybind