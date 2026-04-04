#pragma once

#include <variant>

namespace ginfer::utils {

template <class... Ts>
struct overloaded : Ts... {
  using Ts::operator()...;
};

#define MATCH(var, ...) std::visit(ginfer::utils::overloaded{__VA_ARGS__}, var)

}  // namespace ginfer::utils