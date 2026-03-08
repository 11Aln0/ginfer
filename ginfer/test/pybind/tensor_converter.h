#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <array>
#include <cstddef>
#include "ginfer/common/errors.h"
#include "ginfer/core/tensor/tensor.h"
#include "ginfer/test/pybind/type.h"

namespace ginfer::test::pybind {

namespace py = pybind11;

using common::DeviceType;
using core::memory::Buffer;
using core::tensor::DataType;
using core::tensor::dTypeSize;
using core::tensor::Layout;
using core::tensor::Shape;
using core::tensor::Tensor;
using core::tensor::TensorRef;

class NumpyToTensorConverter {
 public:
  static TensorRef convert(py::array arr) {
    size_t ndim = arr.ndim();
    std::vector<int64_t> shape;
    for (int i = 0; i < ndim; ++i) {
      shape.push_back(arr.shape(i));
    }

    DataType dtype = type::numpyDtypeToTensorDtype(arr.dtype());
    int64_t bytes = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int64_t>()) *
                    dTypeSize(dtype);
    DECLARE_OR_THROW(buf,
                     Buffer::create(bytes, (std::byte*)arr.mutable_data(), DeviceType::kDeviceCPU));

    if ((bool)(arr.flags() & py::array::f_style)) {
      std::reverse(shape.begin(), shape.end());
      DECLARE_OR_THROW(t, Tensor::create(dtype, Shape(shape), buf));
      std::vector<size_t> new_order(ndim);
      for (size_t i = 0; i < ndim; ++i) {
        new_order[i] = ndim - 1 - i;
      }
      return t->permute(new_order);
    } else {
      DECLARE_OR_THROW(t, Tensor::create(dtype, Shape(shape), buf));
      return t;
    }
  }

  static py::array convert_back(const TensorRef tensor) {
    const Shape& shape = tensor->shape();
    std::vector<size_t> np_shape;
    for (size_t i = 0; i < shape.ndim(); ++i) {
      np_shape.push_back(shape[i]);
    }

    py::dtype np_dtype = type::tensorDtypeToNumpyDtype(tensor->dtype());
    py::array np_array(np_dtype, np_shape);
    size_t nbytes = tensor->nbytes();
    std::memcpy(np_array.mutable_data(), tensor->data<void>(), nbytes);
    return np_array;
  }
};

}  // namespace ginfer::test::pybind