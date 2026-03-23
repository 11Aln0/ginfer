#pragma once
#include "ginfer/core/op/kernels/kernel_registry.h"
#include "ginfer/core/tensor/dtype.h"

namespace ginfer::core::op::kernel {

template <typename KernelFuncType>
class KernelDispatcher {
 public:
  explicit KernelDispatcher(std::string kernel_name) : kernel_name_(std::move(kernel_name)) {}

  KernelFuncType getKernel(common::DeviceType dev_type,
                           tensor::DataType in_dtype,
                           tensor::DataType out_dtype) {
    if (has_cached_kernel_ && cached_dev_type_ == dev_type && cached_dtype_.first == in_dtype &&
        cached_dtype_.second == out_dtype) {
      return func_type_;
    }
    return getAndCacheKernel(dev_type, in_dtype, out_dtype);
  }

  KernelFuncType getKernel(common::DeviceType dev_type, tensor::DataType dtype) {
    if (has_cached_kernel_ && cached_dev_type_ == dev_type && cached_dtype_.first == dtype &&
        cached_dtype_.second == dtype) {
      return func_type_;
    }
    return getAndCacheKernel(dev_type, dtype, dtype);
  }

 private:
  KernelFuncType getAndCacheKernel(common::DeviceType dev_type,
                                   tensor::DataType in_dtype,
                                   tensor::DataType out_dtype) {
    auto kernel = KernelRegistry::getInstance(dev_type)->getKernel<KernelFuncType>(
        kernel_name_, in_dtype, out_dtype);
    cached_dtype_ = {in_dtype, out_dtype};
    cached_dev_type_ = dev_type;
    func_type_ = kernel;
    has_cached_kernel_ = true;
    return func_type_;
  }

 private:
  std::pair<tensor::DataType, tensor::DataType> cached_dtype_ = {
      tensor::DataType::kDataTypeVoid, tensor::DataType::kDataTypeVoid};
  common::DeviceType cached_dev_type_ = common::DeviceType::kDeviceUnknown;
  KernelFuncType func_type_ = nullptr;
  bool has_cached_kernel_ = false;
  std::string kernel_name_;
};

}  // namespace ginfer::core::op::kernel