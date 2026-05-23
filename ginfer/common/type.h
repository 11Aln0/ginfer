#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <bit>
#include <cstdint>
#include <iostream>
#include <type_traits>
#include "ginfer/common/device.h"

namespace ginfer::type {

using common::DeviceType;

struct Float16 {
  uint16_t data;
};

struct BFloat16 {
  uint16_t data;
};

using Float32 = float;
using Float64 = double;
using Int64 = int64_t;
using Int32 = int32_t;
using Int8 = int8_t;

template <DeviceType Device, typename T>
struct DeviceNativeTypeOf;

template <DeviceType Device>
struct DeviceNativeTypeOf<Device, Float32> {
  using type = float;
};

template <DeviceType Device>
struct DeviceNativeTypeOf<Device, Float64> {
  using type = double;
};

template <DeviceType Device>
struct DeviceNativeTypeOf<Device, Int64> {
  using type = int64_t;
};

template <DeviceType Device>
struct DeviceNativeTypeOf<Device, Int32> {
  using type = int32_t;
};

template <DeviceType Device>
struct DeviceNativeTypeOf<Device, Int8> {
  using type = int8_t;
};

template <>
struct DeviceNativeTypeOf<DeviceType::kDeviceCUDA, Float16> {
  using type = __half;
};

template <>
struct DeviceNativeTypeOf<DeviceType::kDeviceCPU, Float16> {
  using type = void;  // no native float16 type on CPU
};

template <>
struct DeviceNativeTypeOf<DeviceType::kDeviceCUDA, BFloat16> {
  using type = __nv_bfloat16;
};

template <>
struct DeviceNativeTypeOf<DeviceType::kDeviceCPU, BFloat16> {
  using type = void;  // no native bfloat16 type on CPU
};

template <DeviceType Device, typename T>
struct TypeOf;

template <DeviceType Device>
struct TypeOf<Device, float> {
  using type = Float32;
};

template <DeviceType Device>
struct TypeOf<Device, double> {
  using type = Float64;
};

template <DeviceType Device>
struct TypeOf<Device, int64_t> {
  using type = Int64;
};

template <DeviceType Device>
struct TypeOf<Device, int32_t> {
  using type = Int32;
};

template <DeviceType Device>
struct TypeOf<Device, int8_t> {
  using type = Int8;
};

template <>
struct TypeOf<DeviceType::kDeviceCUDA, __half> {
  using type = Float16;
};

template <>
struct TypeOf<DeviceType::kDeviceCUDA, __nv_bfloat16> {
  using type = BFloat16;
};

template <typename SrcType, typename DstType>
struct TypeConverter {
  static DstType convert(SrcType value) {
    static_assert(std::is_arithmetic_v<SrcType> && std::is_arithmetic_v<DstType>,
                  "TypeConverter default conversion only supports arithmetic types");
    return static_cast<DstType>(value);
  }
};

template <>
struct TypeConverter<Float16, Float32> {
  static Float32 convert(Float16 value) { return convertImpl(value); }

 private:
  static Float32 convertImpl(Float16 value) {
    const uint32_t sign = (static_cast<uint32_t>(value.data) & 0x8000U) << 16U;
    int32_t exponent = static_cast<int32_t>((static_cast<uint32_t>(value.data) >> 10U) & 0x1FU);
    uint32_t fraction = static_cast<uint32_t>(value.data) & 0x03FFU;

    if (exponent == 0) {
      if (fraction == 0U) {
        return std::bit_cast<Float32>(sign);
      }
      exponent = -14;
      while ((fraction & 0x0400U) == 0U) {
        fraction <<= 1U;
        --exponent;
      }
      fraction &= 0x03FFU;
    } else if (exponent == 0x1F) {
      const uint32_t bits = sign | 0x7F800000U | (fraction << 13U);
      return std::bit_cast<Float32>(bits);
    } else {
      exponent -= 15;
    }

    const uint32_t bits = sign | (static_cast<uint32_t>(exponent + 127) << 23U) |
                          (fraction << 13U);
    return std::bit_cast<Float32>(bits);
  }
};

template <>
struct TypeConverter<BFloat16, Float32> {
  static Float32 convert(BFloat16 value) { return convertImpl(value); }

 private:
  static Float32 convertImpl(BFloat16 value) {
    return std::bit_cast<Float32>(static_cast<uint32_t>(value.data) << 16U);
  }
};

}  // namespace ginfer::type
