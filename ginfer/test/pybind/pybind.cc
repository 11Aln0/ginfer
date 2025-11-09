#include "ginfer/test/pybind/pybind.h"

namespace ginfer::test::pybind {

PYBIND11_MODULE(ginfer_test, m) {
  m.doc() = "Pybind11 test for ginfer::op::AddLayer on CUDA";
  m.def("run_add_layer_cuda_test", WRAP_TENSOR_FUNC(test_add_layer_cuda), "Run AddLayer on CUDA using pybind interface");
  m.def("run_rmsnorm_layer_cuda_test", WRAP_TENSOR_FUNC(test_rmsnorm_layer_cuda), "Run RMSNormLayer on CUDA using pybind interface");
}

}  // namespace ginfer::test::pybind
