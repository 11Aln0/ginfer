#pragma once

#include "ginfer/common/type.h"

namespace ginfer::test::pybind::type {

template <typename T>
struct NumpyStorageType {
  using type = T;
};

template <>
struct NumpyStorageType<ginfer::type::Float16> {
  using type = uint16_t;
};

}  // namespace ginfer::test::pybind::type

namespace pybind11::detail {
template <>
struct npy_format_descriptor<ginfer::type::Float16> {
  static constexpr auto name = "float16";
  static pybind11::dtype dtype() { return pybind11::dtype("float16"); }
  static std::string format() { return "e"; }
};

}  // namespace pybind11::detail