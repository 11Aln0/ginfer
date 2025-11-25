#pragma once
#include <pybind11/pybind11.h>
#include <unordered_map>
#include "ginfer/test/pybind/func_wrap.h"

namespace ginfer::test::pybind {

namespace py = pybind11;

class PybindTestRegistry {
 private:
  using PybindTestFunc = py::object (*)(py::args);

  std::unordered_map<std::string, PybindTestFunc> registry_;

 public:
  static PybindTestRegistry* getInstance() {
    static PybindTestRegistry instance;
    return &instance;
  }

  void registerTest(const std::string& name, PybindTestFunc func) { registry_[name] = func; }

  PybindTestFunc getTest(const std::string& name) {
    auto it = registry_.find(name);
    if (it != registry_.end()) {
      return it->second;
    }
    return nullptr;
  }

  auto begin() { return registry_.begin(); }
  auto end() { return registry_.end(); }
  auto begin() const { return registry_.begin(); }
  auto end() const { return registry_.end(); }
};

#define REGISTER_PYBIND_TEST(func)                                                                          \
  namespace {                                                                                               \
  struct PybindTest_##func {                                                                                \
    PybindTest_##func() {                                                                                   \
      ginfer::test::pybind::PybindTestRegistry::getInstance()->registerTest(#func, WRAP_TENSOR_FUNC(func)); \
    }                                                                                                       \
  };                                                                                                        \
  static PybindTest_##func pybind_test_instance_##func;                                                     \
  }

}  // namespace ginfer::test::pybind