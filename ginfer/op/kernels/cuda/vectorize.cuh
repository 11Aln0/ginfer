#pragma once

#include "ginfer/op/kernels/vectorize.h"

#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace ginfer::op::kernel {

template <typename T, int size>
__device__ __forceinline__ AlignedVector<T, size> operator+(const AlignedVector<T, size>& a, const AlignedVector<T, size>& b) {
  AlignedVector<T, size> result;
  #pragma unroll
  for (int i = 0; i < size; ++i) {
    result.val[i] = a.val[i] + b.val[i];
  }
  return result;
}

template <int size>
__device__ __forceinline__ AlignedVector<__half, size> operator+(const AlignedVector<__half, size>& a, const AlignedVector<__half, size>& b) {
  AlignedVector<__half, size> result;
  
#pragma unroll
  for(int i = 0; i < size; i += 2) {
    __half2 va = *reinterpret_cast<const __half2*>(&a.val[i]);
    __half2 vb = *reinterpret_cast<const __half2*>(&b.val[i]);
    __half2 vr = __hadd2(va, vb); // simd
    *reinterpret_cast<__half2*>(&result.val[i]) = vr;
  }
  return result;
}

template <int size>
__device__ __forceinline__ AlignedVector<__nv_bfloat16, size> operator+(const AlignedVector<__nv_bfloat16, size>& a, const AlignedVector<__nv_bfloat16, size>& b) {
  AlignedVector<__nv_bfloat16, size> result;
#pragma unroll
  for(int i = 0; i < size; i += 2) {
    __nv_bfloat162 va = *reinterpret_cast<const __nv_bfloat162*>(&a.val[i]);
    __nv_bfloat162 vb = *reinterpret_cast<const __nv_bfloat162*>(&b.val[i]);
    __nv_bfloat162 vr = __hadd2(va, vb); // simd
    *reinterpret_cast<__nv_bfloat162*>(&result.val[i]) = vr;
  }
  return result;
}


template <typename T>
struct AccType {
  using type = T;
};

template <>
struct AccType<half> {
  using type = float;
};

template <typename T, int size>
struct DotProduct {
  __device__ __forceinline__ static float run(const AlignedVector<T, size>& a, const AlignedVector<T, size>& b) {
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < size; ++i) {
      sum += static_cast<float>(a.val[i]) * static_cast<float>(b.val[i]);
    }
    return sum;
  }
};

// template <int size>
// struct DotProduct<half, size> {
//   __device__ __forceinline__ static float run(const AlignedVector<half, size>& a, const AlignedVector<half, size>& b) {
//     float sum = 0.0f;
// #pragma unroll
//     for(int i = 0; i < size; i += 2) {
//       __half2 va = *reinterpret_cast<const __half2*>(&a.val[i]);
//       __half2 vb = *reinterpret_cast<const __half2*>(&b.val[i]);
//       __half2 vmul = __hmul2(va, vb);
//       sum += __half2float(vmul.x) + __half2float(vmul.y);
//     }
//     return sum;
//   }
// };

// template <int size>
// struct DotProduct<__nv_bfloat16, size> {
//   __device__ __forceinline__ static float run(const AlignedVector<__nv_bfloat16, size>& a, const AlignedVector<__nv_bfloat16, size>& b) {
//     float sum = 0.0f;
// #pragma unroll
//     for(int i = 0; i < size; i += 2) {
//       __nv_bfloat162 va = *reinterpret_cast<const __nv_bfloat162*>(&a.val[i]);
//       __nv_bfloat162 vb = *reinterpret_cast<const __nv_bfloat162*>(&b.val[i]);
//       __nv_bfloat162 vmul = __hmul2(va, vb);
//       sum += __bfloat162float(vmul.x) + __bfloat162float(vmul.y);
//     }
//     return sum;
//   }
// };



}  // namespace ginfer::op::kernel