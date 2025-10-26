#pragma once

#include <cuda_fp16.h>
#include <cstdint>
#include "ginfer/common/device.h"

namespace ginfer::tensor {

enum class DType : uint8_t {
  kDTypeUnknown = 0,
  kDTypeFloat32 = 1,
  kDTypeFloat16 = 2,
  kDTypeInt32 = 3,
  kDTypeInt8 = 4,
};

constexpr size_t dTypeSize(DType dtype) {
  switch (dtype) {
    case DType::kDTypeFloat32:
      return 4;
    case DType::kDTypeFloat16:
      return 2;
    case DType::kDTypeInt32:
      return 4;
    case DType::kDTypeInt8:
      return 1;
    default:
      return 0;
  }
}

template <common::DeviceType dev_type, tensor::DType dtype>
struct DeviceDtype {
  using type = void;
};

template <common::DeviceType dev_type>
struct DeviceDtype<dev_type, tensor::DType::kDTypeFloat32> {
  using type = float;
};

template <common::DeviceType dev_type>
struct DeviceDtype<dev_type, tensor::DType::kDTypeInt32> {
  using type = int32_t;
};

template <common::DeviceType dev_type>
struct DeviceDtype<dev_type, tensor::DType::kDTypeInt8> {
  using type = int8_t;
};

template <>
struct DeviceDtype<common::DeviceType::kDeviceCUDA, tensor::DType::kDTypeFloat16> {
  using type = __half;
};
}  // namespace ginfer::tensor