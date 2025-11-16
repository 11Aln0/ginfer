#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <array>
#include <functional>
#include <tuple>
#include <type_traits>
#include "ginfer/tensor/traits.h"
#include "ginfer/test/pybind/tensor_converter.h"

namespace ginfer::test::pybind {

// function traits
template <typename F>
struct FuncTraits;

template <typename R, typename... Args>
struct FuncTraits<R(Args...)> {
  using ReturnType = R;
  using ArgsTuple = std::tuple<Args...>;
  static constexpr size_t arity = sizeof...(Args);

  template <size_t I>
  using ArgsType = std::decay_t<std::tuple_element_t<I, ArgsTuple>>;
};

template <typename R, typename... Args>
struct FuncTraits<R (*)(Args...)> : FuncTraits<R(Args...)> {};

template <typename R, typename... Args>
struct FuncTraits<std::function<R(Args...)>> : FuncTraits<R(Args...)> {};

template <typename F>
struct FuncTraits : FuncTraits<decltype(&F::operator())> {};

template <typename C, typename R, typename... Args>
struct FuncTraits<R (C::*)(Args...) const> : FuncTraits<R(Args...)> {};

template <typename C, typename R, typename... Args>
struct FuncTraits<R (C::*)(Args...)> : FuncTraits<R(Args...)> {};

namespace py = pybind11;

using PyArgType = decltype(std::declval<py::args>()[0]);

template <typename T>
struct ArgConverter {
  static T convert(const PyArgType& arg) { return arg.cast<T>(); }
};

template <typename T>
struct ReturnValConverter {
  static py::object convert(const T& val) { return py::cast(val); }
};

template <>
struct ArgConverter<tensor::Tensor> {
  static tensor::Tensor convert(const PyArgType& arg) { return NumpyToTensorConverter::convert(arg.cast<py::array>()); }
};

template <>
struct ReturnValConverter<tensor::Tensor> {
  static py::object convert(const tensor::Tensor& val) { return NumpyToTensorConverter::convert_back(val); }
};

template <typename F>
class TensorFuncBridge {
 public:
  using Traits = FuncTraits<F>;
  using ReturnType = typename Traits::ReturnType;

  static constexpr size_t args_cnt = Traits::arity;

  static py::object call(F fn, py::args args) {
    if (args.size() != args_cnt) {
      throw std::runtime_error("Parameter count mismatch");
    }
    return ReturnValConverter<ReturnType>::convert(callImpl(fn, args, std::make_index_sequence<args_cnt>{}));
  }

 private:
  template <size_t... Indices>
  static ReturnType callImpl(F fn, py::args& args, std::index_sequence<Indices...>) {
    auto params = std::make_tuple(ArgConverter<typename Traits::template ArgsType<Indices>>::convert(args[Indices])...);
    return std::apply(fn, params);
  }
};

#define WRAP_TENSOR_FUNC(fn) [](py::args args) { return ginfer::test::pybind::TensorFuncBridge<decltype(&fn)>::call(&fn, args); }

}  // namespace ginfer::test::pybind