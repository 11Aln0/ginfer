#pragma once
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace ginfer::test::pybind {

py::object run_add_layer_cuda_test(py::array a_np, py::array b_np);

py::object run_rmsnorm_layer_cuda_test(py::array input_np, py::array gamma_np, float epsilon);

}  // namespace ginfer::test::pybind