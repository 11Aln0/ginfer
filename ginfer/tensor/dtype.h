#pragma once

#include <cuda_fp16.h>
#include <cstdint>
#include <iostream>
#include "ginfer/common/device.h"

namespace ginfer::tensor {

enum class Dtype : uint8_t {
  kDtypeUnknown = 0,
  kDtypeFloat32 = 1,
  kDtypeFloat16 = 2,
  kDtypeInt32 = 3,
  kDtypeInt8 = 4,
};

constexpr size_t dTypeSize(Dtype dtype) {
  switch (dtype) {
    case Dtype::kDtypeFloat32:
      return 4;
    case Dtype::kDtypeFloat16:
      return 2;
    case Dtype::kDtypeInt32:
      return 4;
    case Dtype::kDtypeInt8:
      return 1;
    default:
      return 0;
  }
}

template <common::DeviceType dev_type, tensor::Dtype dtype>
struct DeviceDtype {
  using type = void;
};

template <common::DeviceType dev_type>
struct DeviceDtype<dev_type, tensor::Dtype::kDtypeFloat32> {
  using type = float;
};

template <common::DeviceType dev_type>
struct DeviceDtype<dev_type, tensor::Dtype::kDtypeInt32> {
  using type = int32_t;
};

template <common::DeviceType dev_type>
struct DeviceDtype<dev_type, tensor::Dtype::kDtypeInt8> {
  using type = int8_t;
};

template <>
struct DeviceDtype<common::DeviceType::kDeviceCUDA, tensor::Dtype::kDtypeFloat16> {
  using type = __half;
};
}  // namespace ginfer::tensor