#pragma once

#include <memory>

#include "ginfer/memory/allocator.h"
#include "ginfer/memory/cuda/cuda_allocator.h"

namespace ginfer::memory {
template <class T, typename = typename std::enable_if_t<std::is_base_of<DeviceAllocator, T>::value>>
class DeviceAllocatorFactory {
 public:
  DeviceAllocatorFactory() = delete;
  DeviceAllocatorFactory(const DeviceAllocatorFactory&) = delete;
  DeviceAllocatorFactory& operator=(const DeviceAllocatorFactory&) = delete;

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
using CUDAAllocatorFactory = DeviceAllocatorFactory<cuda::CUDADeviceAllocator<S>>;
using CPUAllocatorFactory = DeviceAllocatorFactory<CPUDeviceAllocator>;

using DefaultCUDAAllocatorFactory = CUDAAllocatorFactory<cuda::DefaultAllocStrategy>;
using DefaultCPUAllocatorFactory = CPUAllocatorFactory;

DeviceAllocator* getDefaultDeviceAllocator(DeviceType dev_type);
}  // namespace ginfer::memory