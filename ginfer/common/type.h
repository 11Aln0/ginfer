#pragma once

#include <cuda_fp16.h>
#include <cstdint>
#include <iostream>
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

}  // namespace ginfer::type