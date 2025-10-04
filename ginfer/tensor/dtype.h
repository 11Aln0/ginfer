#pragma once

#include <cstdint>

namespace ginfer::tensor {

enum class DType : uint8_t {
  kDTypeUnknown = 0,
  kDTypeFloat32 = 1,
  kDtypeFloat16 = 2,
  kDTypeInt32 = 3,
  kDTypeInt8 = 4,
};

constexpr size_t dTypeSize(DType dtype) {
  switch (dtype) {
    case DType::kDTypeFloat32:
      return 4;
    case DType::kDtypeFloat16:
      return 2;
    case DType::kDTypeInt32:
      return 4;
    case DType::kDTypeInt8:
      return 1;
    default:
      return 0;
  }
}

}  // namespace ginfer::tensor