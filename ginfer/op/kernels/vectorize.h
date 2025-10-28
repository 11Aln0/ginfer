#pragma once

namespace ginfer::op::kernel {

template <typename T>
struct DefaultVecSize {
  static constexpr size_t value = 16 / sizeof(T);  // 128 bits
};

template <typename T, size_t size>
struct alignas(sizeof(T) * size) AlignedVector {
  static_assert(size > 1 && (size & (size - 1)) == 0, 
              "size must be a power of 2 and greater than 1");
  static_assert(size <= DefaultVecSize<T>::value, 
              "size exceeds the maximum vector size for type T");
  T val[size];
};

}  // namespace ginfer::op::kernel