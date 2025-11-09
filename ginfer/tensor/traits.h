#pragma once
#include <type_traits>
#include "ginfer/tensor/tensor.h"

namespace ginfer::tensor {

template <typename T>
struct IsTensor : std::false_type {};

template <>
struct IsTensor<Tensor> : std::true_type {};

template <>
struct IsTensor<const Tensor> : std::true_type {};

template <typename T>
struct IsTensor<T*> : IsTensor<T> {};

template <typename T>
struct IsTensor<T&> : IsTensor<T> {};

}  // namespace ginfer::tensor