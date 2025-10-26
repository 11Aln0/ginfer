#pragma once

#include <cuda_fp16.h>

namespace ginfer::op::kernel {

template <typename T>
__device__ __forceinline__ T operator+(const T& a, const T& b) {
  return a + b;
}

template <>
__device__ __forceinline__ __half operator+<__half>(const __half& a, const __half& b) {
  return __hadd(a, b);
};

}  // namespace ginfer::op::kernel