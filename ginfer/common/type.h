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

using Float32 = float;
using Float64 = double;
using Int32 = int32_t;
using Int8 = int8_t;

template <typename T>
struct NumpyType {
  using type = T;
};

template <>
struct NumpyType<Float16> {
  using type = uint16_t;
};

template <DeviceType dev_type, typename T>
struct DeviceNativeTypeOf {
  using type = T;
};

template <>
struct DeviceNativeTypeOf<DeviceType::kDeviceCUDA, Float16> {
  using type = __half;
};

template <>
struct DeviceNativeTypeOf<DeviceType::kDeviceCPU, Float16> {
  using type = void;  // no native float16 type on CPU
};

}  // namespace ginfer::type