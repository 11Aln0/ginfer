#pragma once

#include <memory>

#include "ginfer/memory/allocator.h"
#include "ginfer/memory/cuda/cuda_allocator.h"

namespace ginfer::memory {
template <class T, typename = typename std::enable_if_t<std::is_base_of<DeviceAllocator, T>::value>>
class GlobalDeviceAllocator {
 public:
  GlobalDeviceAllocator() = delete;
  GlobalDeviceAllocator(const GlobalDeviceAllocator&) = delete;
  GlobalDeviceAllocator& operator=(const GlobalDeviceAllocator&) = delete;

  static T* get_instance() {
    if (instance == nullptr) {
      instance = new T();
    }
    return instance;
  }

 private:
  inline static T* instance = nullptr;
};

template <typename S>
using GlobalCUDAAllocator = GlobalDeviceAllocator<cuda::CUDADeviceAllocator<S>>;
using GlobalCPUAllocator = GlobalDeviceAllocator<CPUDeviceAllocator>;

using DefaultGlobalCUDAAllocator = GlobalCUDAAllocator<cuda::DefaultAllocStrategy>;
using DefaultGlobalCPUAllocator = GlobalCPUAllocator;

DeviceAllocator* getDefaultDeviceAllocator(DeviceType dev_type);
}  // namespace ginfer::memory