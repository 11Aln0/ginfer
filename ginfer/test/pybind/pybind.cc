#include "ginfer/test/pybind/pybind.h"
#include "ginfer/test/pybind/func_wrap.h"
#include "ginfer/test/pybind/test_registry.h"

namespace ginfer::test::pybind {

PYBIND11_MODULE(ginfer_test, m) {
  m.doc() = "Pybind11 test for ginfer::core::op::AddLayer on CUDA";
  for (const auto& [name, func] : *PybindTestRegistry::getInstance()) {
    m.def(name.c_str(), func);
  }
}

}  // namespace ginfer::test::pybind
