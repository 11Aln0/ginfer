#pragma once

#include <functional>
#include <vector>
#include "ginfer/common/device.h"
#include "ginfer/core/tensor/dtype.h"
#include "ginfer/core/tensor/tensor.h"

namespace ginfer::core::op::kernel {

template <common::DeviceType>
struct KernelFuncType {
  using type = void;
};

template <>
struct KernelFuncType<common::DeviceType::kDeviceCPU> {
  using type = void (*)(const std::vector<const tensor::Tensor*>& inputs, tensor::Tensor* output);
};

template <>
struct KernelFuncType<common::DeviceType::kDeviceCUDA> {
  using type = void (*)(const std::vector<const tensor::Tensor*>& inputs,
                        tensor::Tensor* output,
                        void* stream);
};

struct KernelInfo {
  std::string name;
  common::DeviceType dev_type;
  tensor::DataType input_dtype;
  tensor::DataType output_dtype;

  KernelInfo()
      : name("unknown_kernel"),
        dev_type(common::DeviceType::kDeviceUnknown),
        input_dtype(tensor::DataType::kDataTypeVoid),
        output_dtype(tensor::DataType::kDataTypeVoid) {}

  KernelInfo(const std::string& n,
             tensor::DataType in_type,
             tensor::DataType out_type,
             common::DeviceType device_type)
      : name(n), dev_type(device_type), input_dtype(in_type), output_dtype(out_type) {}

  bool operator==(const KernelInfo& other) const {
    return name == other.name && input_dtype == other.input_dtype &&
           output_dtype == other.output_dtype && dev_type == other.dev_type;
  }
};

}  // namespace ginfer::core::op::kernel

namespace std {

template <>
struct hash<ginfer::core::op::kernel::KernelInfo> {
  size_t operator()(const ginfer::core::op::kernel::KernelInfo& k) const;

 private:
  template <class T>
  static void hash_combine(size_t& seed, const T& v) {
    seed ^= hash<T>{}(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  }
};

}  // namespace std