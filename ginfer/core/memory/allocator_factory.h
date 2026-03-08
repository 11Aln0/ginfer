#pragma once

#include <stdexcept>

#include "ginfer/core/memory/alloc_strategy.h"
#include "ginfer/core/memory/allocator.h"

namespace ginfer::core::memory {

template <class T, typename = typename std::enable_if_t<std::is_base_of<DeviceAllocator, T>::value>>
class GlobalDeviceAllocator {
 public:
  GlobalDeviceAllocator() = delete;
  GlobalDeviceAllocator(const GlobalDeviceAllocator&) = delete;
  GlobalDeviceAllocator& operator=(const GlobalDeviceAllocator&) = delete;

  static T* getInstance() {
    if (instance == nullptr) {
      instance = new T();
    }
    return instance;
  }

 private:
  inline static T* instance = nullptr;
};

using GlobalCUDAAllocator = GlobalDeviceAllocator<CUDADeviceAllocator>;
using GlobalCPUAllocator = GlobalDeviceAllocator<CPUDeviceAllocator>;

using DefaultGlobalCUDAAllocator = GlobalCUDAAllocator;
using DefaultGlobalCPUAllocator = GlobalCPUAllocator;

template <template <typename> class Strategy = DefaultAlllocStrategy>
DeviceAllocator* getDeviceAllocator(DeviceType dev_type) {
  switch (dev_type) {
    case DeviceType::kDeviceCPU:
      return GlobalDeviceAllocator<Strategy<CPUDeviceAllocator>>::getInstance();
    case DeviceType::kDeviceCUDA:
      return GlobalDeviceAllocator<Strategy<CUDADeviceAllocator>>::getInstance();
    default:
      throw std::invalid_argument("Unsupported device type.");
  }
}

inline DeviceAllocator* getDefaultDeviceAllocator(DeviceType dev_type) {
  return getDeviceAllocator(dev_type);
}

}  // namespace ginfer::core::memory
