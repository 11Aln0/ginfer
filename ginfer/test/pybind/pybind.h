#pragma once
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include "ginfer/test/pybind/func_wrap.h"

namespace py = pybind11;

namespace ginfer::test::pybind {

Tensor test_add_layer_cuda(Tensor& a_tensor, Tensor& b_tensor);

Tensor test_rmsnorm_layer_cuda(Tensor& input_tensor, Tensor& gamma_tensor, float epsilon);

}  // namespace ginfer::test::pybind