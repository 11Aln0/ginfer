#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <iostream>
#include "ginfer/common/device.h"
#include "ginfer/common/type.h"

namespace ginfer::tensor {

enum class DataType : uint8_t {
  kDataTypeUnknown = 0,
  kDataTypeFloat32 = 1,
  kDataTypeFloat16 = 2,
  kDataTypeBFloat16 = 3,
  kDataTypeInt64 = 4,
  kDataTypeInt32 = 5,
  kDataTypeInt8 = 6,
};

inline constexpr size_t dTypeSize(DataType dtype) {
  switch (dtype) {
    case DataType::kDataTypeFloat32:
      return 4;
    case DataType::kDataTypeFloat16:
      return 2;
    case DataType::kDataTypeBFloat16:
      return 2;
    case DataType::kDataTypeInt64:
      return 8;
    case DataType::kDataTypeInt32:
      return 4;
    case DataType::kDataTypeInt8:
      return 1;
    default:
      return 0;
  }
}

// mapping from tensor DataType to ginfer type
template <DataType dtype>
struct TypeOf;

template <>
struct TypeOf<DataType::kDataTypeFloat32> {
  using type = type::Float32;
};

template <>
struct TypeOf<DataType::kDataTypeFloat16> {
  using type = type::Float16;
};

template <>
struct TypeOf<DataType::kDataTypeBFloat16> {
  using type = type::BFloat16;
};

template <>
struct TypeOf<DataType::kDataTypeInt64> {
  using type = type::Int64;
};

template <>
struct TypeOf<DataType::kDataTypeInt32> {
  using type = type::Int32;
};

template <>
struct TypeOf<DataType::kDataTypeInt8> {
  using type = type::Int8;
};

// mapping from ginfer type to tensor DataType
template <typename T>
struct DataTypeOf;

template <>
struct DataTypeOf<type::Float32> {
  static constexpr DataType dtype = DataType::kDataTypeFloat32;
};

template <>
struct DataTypeOf<type::Float16> {
  static constexpr DataType dtype = DataType::kDataTypeFloat16;
};

template <>
struct DataTypeOf<type::BFloat16> {
  static constexpr DataType dtype = DataType::kDataTypeBFloat16;
};

template <>
struct DataTypeOf<type::Int64> {
  static constexpr DataType dtype = DataType::kDataTypeInt64;
};

template <>
struct DataTypeOf<type::Int32> {
  static constexpr DataType dtype = DataType::kDataTypeInt32;
};

template <>
struct DataTypeOf<type::Int8> {
  static constexpr DataType dtype = DataType::kDataTypeInt8;
};

}  // namespace ginfer::tensor