#include "ginfer/test/pybind/pybind.h"

namespace ginfer::test {

PYBIND11_MODULE(ginfer_test, m) {
  m.doc() = "Pybind11 test for ginfer::op::AddLayer on CUDA";
  m.def("run_add_layer_cuda_test", &ginfer::test::run_add_layer_cuda_test,
        "Run AddLayer on CUDA using pybind interface");
}

}  // namespace ginfer::test
