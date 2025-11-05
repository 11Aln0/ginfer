#pragma once

#include "ginfer/common/device.h"
#include "ginfer/op/kernels/kernel.h"
#include "ginfer/tensor/dtype.h"
#include "ginfer/utils/macros.h"

#include <glog/logging.h>
#include <memory>
#include <typeinfo>
#include <unordered_map>

namespace ginfer::op::kernel {

using ginfer::common::DeviceType;

class KernelRegistry {
 public:
  inline static std::unordered_map<DeviceType, std::shared_ptr<KernelRegistry>> registries_ = {
      {DeviceType::kDeviceCPU, nullptr}, {DeviceType::kDeviceCUDA, nullptr}};

  static std::shared_ptr<KernelRegistry> getInstance(DeviceType dev_type) {
    if (registries_[dev_type] == nullptr) {
      registries_[dev_type] = std::make_shared<KernelRegistry>(dev_type);
    }
    return registries_[dev_type];
  }

  KernelRegistry(DeviceType dev_type) : dev_type_(dev_type), kernels_({}) {}

  DeviceType getDeviceType() const { return dev_type_; }

  template <typename FuncType>
  void registerKernel(const KernelInfo& kernel_info, FuncType func) {
    LOG(INFO) << "Registering kernel: " << kernel_info.name << " on device: " << dev_type_;
    kernels_[kernel_info] = {reinterpret_cast<void*>(func), typeid(FuncType).hash_code()};
  }

  template <typename FuncType>
  FuncType getKernel(const KernelInfo& kernel_info) {
    auto it = kernels_.find(kernel_info);
    CHECK(it != kernels_.end()) << "Kernel not found: " << kernel_info.name;
    const KernelRegistryEntry& entry = it->second;
    CHECK(entry.type == typeid(FuncType).hash_code())
        << "Kernel function type mismatch for kernel: " << kernel_info.name;
    return reinterpret_cast<FuncType>(entry.func_ptr);
  }

  template <typename FuncType>
  FuncType getKernel(const std::string& name, tensor::DataType in_dtype,
                     tensor::DataType out_dtype) {
    return getKernel<FuncType>(KernelInfo(name, in_dtype, out_dtype, dev_type_));
  }

  template <typename FuncType>
  FuncType getKernel(const std::string& name, tensor::DataType dtype) {
    return getKernel<FuncType>(KernelInfo(name, dtype, dtype, dev_type_));
  }

 private:
  struct KernelRegistryEntry {
    void* func_ptr;
    size_t type;
  };

  DeviceType dev_type_;
  std::unordered_map<KernelInfo, KernelRegistryEntry> kernels_;
};

#define _REGISTER_KERNEL_FROM_INFO(dev_type, kernel_info, func) \
  ::ginfer::op::kernel::KernelRegistry::getInstance(dev_type)->registerKernel(kernel_info, func);

#define _REGISTER_KERNEL_DIFF_DTYPE(name, dev_type, func, in_dtype, out_dtype) \
  _REGISTER_KERNEL_FROM_INFO(                                                  \
      dev_type, ::ginfer::op::kernel::KernelInfo(#name, in_dtype, out_dtype, dev_type), func)

#define _REGISTER_KERNEL_SAME_DTYPE(name, dev_type, func, dtype) \
  _REGISTER_KERNEL_DIFF_DTYPE(name, dev_type, func, dtype, dtype)

#define INSTANTIATE_KERNEL_FUNC(tpl_func, dev_type, dtype)                        \
  (&tpl_func<typename ::ginfer::type::DeviceNativeTypeOf<                         \
                 dev_type, typename ::ginfer::tensor::TypeOf<dtype>::type>::type, \
             ::ginfer::common::DeviceContext>)

#define _REGISTER_TPL_KERNEL(name, dev_type, tpl_func, dtype)                                     \
  _REGISTER_KERNEL_SAME_DTYPE(name, dev_type, INSTANTIATE_KERNEL_FUNC(tpl_func, dev_type, dtype), \
                              dtype)

#define _REG_TPL_KRNL_DT_1(name, dev_type, tpl_func, dtype) \
  _REGISTER_TPL_KERNEL(name, dev_type, tpl_func, dtype)
#define _REG_TPL_KRNL_DT_2(name, dev_type, tpl_func, dtype, ...) \
  _REGISTER_TPL_KERNEL(name, dev_type, tpl_func, dtype)          \
  _REG_TPL_KRNL_DT_1(name, dev_type, tpl_func, __VA_ARGS__)
#define _REG_TPL_KRNL_DT_3(name, dev_type, tpl_func, dtype, ...) \
  _REGISTER_TPL_KERNEL(name, dev_type, tpl_func, dtype)          \
  _REG_TPL_KRNL_DT_2(name, dev_type, tpl_func, __VA_ARGS__)
#define _REG_TPL_KRNL_DT_4(name, dev_type, tpl_func, dtype, ...) \
  _REGISTER_TPL_KERNEL(name, dev_type, tpl_func, dtype)          \
  _REG_TPL_KRNL_DT_3(name, dev_type, tpl_func, __VA_ARGS__)
#define _REG_TPL_KRNL_DT_5(name, dev_type, tpl_func, dtype, ...) \
  _REGISTER_TPL_KERNEL(name, dev_type, tpl_func, dtype)          \
  _REG_TPL_KRNL_DT_4(name, dev_type, tpl_func, __VA_ARGS__)

#define _GET_REGISTER_KERNEL_MACRO(_1, _2, _3, _4, _5, NAME, ...) NAME

#define CONCAT_KERNEL(a, b, c) a##b##c##Kernel
#define CONCAT_OBJ(a, b, c) a##b##c##KernelObj

#define REGISTER_KERNEL(name, dev_type, tpl_func, ...)                                       \
  namespace {                                                                                \
  struct CONCAT_KERNEL(Register, name, dev_type) {                                           \
    CONCAT_KERNEL(Register, name, dev_type)() {                                              \
      _GET_REGISTER_KERNEL_MACRO(__VA_ARGS__, _REG_TPL_KRNL_DT_5, _REG_TPL_KRNL_DT_4,        \
                                 _REG_TPL_KRNL_DT_3, _REG_TPL_KRNL_DT_2, _REG_TPL_KRNL_DT_1) \
      (name, common::DeviceType::dev_type, tpl_func, __VA_ARGS__);                           \
    }                                                                                        \
  };                                                                                         \
  static CONCAT_KERNEL(Register, name, dev_type) CONCAT_OBJ(register, name, dev_type);       \
  }

}  // namespace ginfer::op::kernel
