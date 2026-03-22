#pragma once

#include "ginfer/common/device.h"
#include "ginfer/core/op/kernels/kernel.h"
#include "ginfer/core/tensor/dtype.h"
#include "ginfer/utils/macros.h"

#include <glog/logging.h>
#include <memory>
#include <typeinfo>
#include <unordered_map>

namespace ginfer::core::op::kernel {

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

  // TODO: constraint for diffent input/output dtype
  template <typename FuncType>
  FuncType getKernel(const std::string& name,
                     tensor::DataType in_dtype,
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

#define _UNWRAP(...) __VA_ARGS__
#define _PAIR_FIRST_IMPL(first, second) first
#define _PAIR_SECOND_IMPL(first, second) second
#define _PAIR_FIRST(pair) _PAIR_FIRST_IMPL pair
#define _PAIR_SECOND(pair) _PAIR_SECOND_IMPL pair

#define _KERNEL_DEVICE_TYPE_IMPL(dev_type) ::ginfer::common::DeviceType::kDevice##dev_type
#define _KERNEL_DEVICE_TYPE(dev_type) _KERNEL_DEVICE_TYPE_IMPL(dev_type)
#define _KERNEL_DATA_TYPE_IMPL(dtype) ::ginfer::core::tensor::DataType::kDataType##dtype
#define _KERNEL_DATA_TYPE(dtype) _KERNEL_DATA_TYPE_IMPL(dtype)

#define _REGISTER_KERNEL_FROM_INFO(dev_type, kernel_info, func)                                  \
  ::ginfer::core::op::kernel::KernelRegistry::getInstance(dev_type)->registerKernel(kernel_info, \
                                                                                    func);

#define _REGISTER_KERNEL_DIFF_DTYPE(name, dev_type, func, in_dtype, out_dtype)                \
  _REGISTER_KERNEL_FROM_INFO(                                                                 \
      dev_type, ::ginfer::core::op::kernel::KernelInfo(#name, in_dtype, out_dtype, dev_type), \
      func)

#define _REGISTER_KERNEL_SAME_DTYPE(name, dev_type, func, dtype) \
  _REGISTER_KERNEL_DIFF_DTYPE(name, dev_type, func, dtype, dtype)

#define _REGISTER_KERNEL_DIFF_IO(name, dev_type, func, in_dtype, out_dtype) \
  _REGISTER_KERNEL_DIFF_DTYPE(name, _KERNEL_DEVICE_TYPE(dev_type), func,    \
                              _KERNEL_DATA_TYPE(in_dtype), _KERNEL_DATA_TYPE(out_dtype))

#define INSTANTIATE_KERNEL_FUNC(tpl_func, dev_type, dtype)                                       \
  (&tpl_func<typename ::ginfer::type::DeviceNativeTypeOf<                                        \
                 _KERNEL_DEVICE_TYPE(dev_type),                                                  \
                 typename ::ginfer::core::tensor::TypeOf<_KERNEL_DATA_TYPE(dtype)>::type>::type, \
             ::ginfer::common::DeviceContext>)

#define INSTANTIATE_KERNEL_FUNC_IO(tpl_func, dev_type, in_dtype, out_dtype)                   \
  (&tpl_func<                                                                                 \
      typename ::ginfer::type::DeviceNativeTypeOf<                                            \
          _KERNEL_DEVICE_TYPE(dev_type),                                                      \
          typename ::ginfer::core::tensor::TypeOf<_KERNEL_DATA_TYPE(in_dtype)>::type>::type,  \
      typename ::ginfer::type::DeviceNativeTypeOf<                                            \
          _KERNEL_DEVICE_TYPE(dev_type),                                                      \
          typename ::ginfer::core::tensor::TypeOf<_KERNEL_DATA_TYPE(out_dtype)>::type>::type, \
      ::ginfer::common::DeviceContext>)

#define INSTANTIATE_KERNEL_FUNC_DIFF_IO(tpl_func, dev_type, in_dtype, out_dtype) \
  INSTANTIATE_KERNEL_FUNC_IO(tpl_func, dev_type, in_dtype, out_dtype)

#define _REGISTER_TPL_KERNEL(name, dev_type, tpl_func, dtype)                                  \
  _REGISTER_KERNEL_DIFF_IO(name, dev_type, INSTANTIATE_KERNEL_FUNC(tpl_func, dev_type, dtype), \
                           dtype, dtype)

#define _REGISTER_TPL_KERNEL_IO(name, dev_type, tpl_func, pair)                                 \
  _REGISTER_KERNEL_DIFF_IO(                                                                      \
      name, dev_type,                                                                            \
      INSTANTIATE_KERNEL_FUNC_IO(tpl_func, dev_type, _PAIR_FIRST(pair), _PAIR_SECOND(pair)),    \
      _PAIR_FIRST(pair), _PAIR_SECOND(pair))

// clang-format off
#define _REG_KRNL_FOREACH_1(action, name, dev_type, tpl_func, x) \
  action(name, dev_type, tpl_func, x)
#define _REG_KRNL_FOREACH_2(action, name, dev_type, tpl_func, x, ...) \
  action(name, dev_type, tpl_func, x)                                  \
  _REG_KRNL_FOREACH_1(action, name, dev_type, tpl_func, __VA_ARGS__)
#define _REG_KRNL_FOREACH_3(action, name, dev_type, tpl_func, x, ...) \
  action(name, dev_type, tpl_func, x)                                  \
  _REG_KRNL_FOREACH_2(action, name, dev_type, tpl_func, __VA_ARGS__)
#define _REG_KRNL_FOREACH_4(action, name, dev_type, tpl_func, x, ...) \
  action(name, dev_type, tpl_func, x)                                  \
  _REG_KRNL_FOREACH_3(action, name, dev_type, tpl_func, __VA_ARGS__)
#define _REG_KRNL_FOREACH_5(action, name, dev_type, tpl_func, x, ...) \
  action(name, dev_type, tpl_func, x)                                  \
  _REG_KRNL_FOREACH_4(action, name, dev_type, tpl_func, __VA_ARGS__)
// clang-format on

#define _GET_REGISTER_KERNEL_FOREACH_MACRO(_1, _2, _3, _4, _5, NAME, ...) NAME
#define _REG_KRNL_FOREACH(action, name, dev_type, tpl_func, ...)                            \
  _GET_REGISTER_KERNEL_FOREACH_MACRO(__VA_ARGS__, _REG_KRNL_FOREACH_5, _REG_KRNL_FOREACH_4, \
                                     _REG_KRNL_FOREACH_3, _REG_KRNL_FOREACH_2,              \
                                     _REG_KRNL_FOREACH_1)                                   \
  (action, name, dev_type, tpl_func, __VA_ARGS__)

#define CONCAT_KERNEL(a, b, c) a##b##c##Kernel
#define CONCAT_OBJ(a, b, c) a##b##c##KernelObj

#define REGISTER_KERNEL(name, dev_type, tpl_func, ...)                                 \
  namespace {                                                                          \
  struct CONCAT_KERNEL(Register, name, dev_type) {                                     \
    CONCAT_KERNEL(Register, name, dev_type)() {                                        \
      _REG_KRNL_FOREACH(_REGISTER_TPL_KERNEL, name, dev_type, tpl_func, __VA_ARGS__);  \
    }                                                                                  \
  };                                                                                   \
  static CONCAT_KERNEL(Register, name, dev_type) CONCAT_OBJ(register, name, dev_type); \
  }

#define REGISTER_KERNEL_DIFF_IO(name, dev_type, tpl_func, ...)                                     \
  namespace {                                                                                      \
  struct CONCAT_KERNEL(RegisterDiffIO, name, dev_type) {                                           \
    CONCAT_KERNEL(RegisterDiffIO, name, dev_type)() {                                              \
      _REG_KRNL_FOREACH(_REGISTER_TPL_KERNEL_IO, name, dev_type, tpl_func, __VA_ARGS__);           \
    }                                                                                              \
  };                                                                                               \
  static CONCAT_KERNEL(RegisterDiffIO, name, dev_type) CONCAT_OBJ(registerDiffIO, name, dev_type); \
  }

}  // namespace ginfer::core::op::kernel
