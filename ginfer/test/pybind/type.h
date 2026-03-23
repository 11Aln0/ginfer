#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <unordered_map>
#include "ginfer/common/type.h"
#include "ginfer/core/tensor/dtype.h"

namespace ginfer::test::pybind::type {

namespace py = pybind11;

template <typename T>
struct NumpyStorageType {
  using type = T;
};

template <>
struct NumpyStorageType<ginfer::type::Float16> {
  using type = uint16_t;
};

using core::tensor::DataType;

inline DataType numpyDtypeToTensorDtype(py::dtype np_dtype) {
  static std::unordered_map<std::string, DataType> dtype_map = {
      {"float32", DataType::kDataTypeFloat32},   {"float16", DataType::kDataTypeFloat16},
      {"bfloat16", DataType::kDataTypeBFloat16}, {"int64", DataType::kDataTypeInt64},
      {"int32", DataType::kDataTypeInt32},       {"int8", DataType::kDataTypeInt8},
  };
  auto it = dtype_map.find(py::str(np_dtype));
  return (it != dtype_map.end()) ? it->second : DataType::kDataTypeVoid;
}

inline py::dtype tensorDtypeToNumpyDtype(DataType dtype) {
  static std::unordered_map<DataType, py::dtype> dtype_map = {
      {DataType::kDataTypeFloat32, py::dtype("float32")},
      {DataType::kDataTypeFloat16, py::dtype("float16")},
      {DataType::kDataTypeBFloat16, py::dtype("bfloat16")},
      {DataType::kDataTypeInt64, py::dtype("int64")},
      {DataType::kDataTypeInt32, py::dtype("int32")},
      {DataType::kDataTypeInt8, py::dtype("int8")},
  };
  auto it = dtype_map.find(dtype);
  return (it != dtype_map.end()) ? it->second : py::dtype("float32");
}

}  // namespace ginfer::test::pybind::type

namespace pybind11::detail {
template <>
struct npy_format_descriptor<ginfer::type::Float16> {
  static constexpr auto name = "float16";
  static pybind11::dtype dtype() { return pybind11::dtype("float16"); }
  static std::string format() { return "e"; }
};

}  // namespace pybind11::detail
