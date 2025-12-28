#pragma once

#include <cuda_fp16.h>

namespace ginfer::op::kernel {

template <typename T>
struct DefaultVecSize {
  static constexpr int value = 16 / sizeof(T);  // 128 bits
};

template <typename T, int size>
struct alignas(sizeof(T) * size) AlignedVector {
  static_assert(size > 1 && (size & (size - 1)) == 0, 
              "size must be a power of 2 and greater than 1");
  static_assert(size <= DefaultVecSize<T>::value, 
              "size exceeds the maximum vector size for type T");
  T val[size];

  __device__ __forceinline__ AlignedVector() {
  #pragma unroll
      for (int i = 0; i < size; i++) val[i] = T(0);
  }
};


}  // namespace ginfer::op::kernel