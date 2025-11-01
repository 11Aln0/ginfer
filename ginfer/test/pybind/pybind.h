#pragma once
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace ginfer::test {

py::object run_add_layer_cuda_test(py::array a_np, py::array b_np);

}  // namespace ginfer::test