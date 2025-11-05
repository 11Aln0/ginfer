#include "ginfer/test/pybind/pybind.h"

namespace ginfer::test::pybind {

PYBIND11_MODULE(ginfer_test, m) {
  m.doc() = "Pybind11 test for ginfer::op::AddLayer on CUDA";
  m.def("run_add_layer_cuda_test", &run_add_layer_cuda_test, "Run AddLayer on CUDA using pybind interface");
  m.def("run_rmsnorm_layer_cuda_test", &run_rmsnorm_layer_cuda_test, "Run RMSNormLayer on CUDA using pybind interface");
}

}  // namespace ginfer::test::pybind
