#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <stdexcept>

#include "ginfer/core/memory/alloc_strategy.h"
#include "ginfer/core/memory/allocator.h"

namespace ginfer::core::memory {

constexpr size_t kDeviceTypeCount = static_cast<size_t>(DeviceType::kDeviceMax);
constexpr size_t kCPUDeviceIndex = static_cast<size_t>(DeviceType::kDeviceCPU);
constexpr size_t kCUDADeviceIndex = static_cast<size_t>(DeviceType::kDeviceCUDA);

enum AllocFlags : uint32_t {
  kDefault = 0,
  kPinned = 1 << 0,
  kPooled = 1 << 1,
  kMaxFlags = 1 << 2,
};

class AllocatorFactory {
 public:
  using Creator = std::function<DeviceAllocator*()>;

  static DeviceAllocator* getAllocator(DeviceType dev_type, uint8_t alloc_flags) {
    return getInstance().get(dev_type, alloc_flags);
  }

 private:
  AllocatorFactory() { initCreators(); }

  static AllocatorFactory& getInstance() {
    static AllocatorFactory instance;
    return instance;
  }

  void initCreators() {
    creators_[kCPUDeviceIndex][kDefault] = [] { return new CPUDeviceAllocator(); };
    creators_[kCPUDeviceIndex][kPinned] = [] { return new PinnedCPUAllocator(); };
    creators_[kCPUDeviceIndex][kPooled] = [] {
      return new PooledAllocStrategy<CPUDeviceAllocator>();
    };
    creators_[kCPUDeviceIndex][kPinned | kPooled] = [] {
      return new PooledAllocStrategy<PinnedCPUAllocator>();
    };

    creators_[kCUDADeviceIndex][kDefault] = [] { return new CUDADeviceAllocator(); };
    creators_[kCUDADeviceIndex][kPooled] = [] {
      return new PooledAllocStrategy<CUDADeviceAllocator>();
    };
  }

  DeviceAllocator* get(DeviceType dev_type, uint8_t alloc_flags) {
    auto dev_idx = static_cast<size_t>(dev_type);
    if (dev_idx >= kDeviceTypeCount) {
      throw std::invalid_argument("Unsupported device type.");
    }
    if (alloc_flags >= kAllocFlagsCount) {
      throw std::invalid_argument("Unsupported allocator flags.");
    }

    auto& allocator = allocators_[dev_idx][alloc_flags];
    auto* instance = allocator.load(std::memory_order_acquire);
    if (instance != nullptr) {
      return instance;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    instance = allocator.load(std::memory_order_acquire);
    if (instance != nullptr) {
      return instance;
    }

    auto& creator = creators_[dev_idx][alloc_flags];
    if (!creator) {
      throw std::invalid_argument("Unsupported allocator flags for device type.");
    }

    instance = creator();
    allocator.store(instance, std::memory_order_release);
    return instance;
  }

 private:
  static constexpr size_t kAllocFlagsCount = kMaxFlags;

  std::atomic<DeviceAllocator*> allocators_[kDeviceTypeCount][kAllocFlagsCount]{};
  Creator creators_[kDeviceTypeCount][kAllocFlagsCount]{};
  std::mutex mutex_;
};

inline DeviceAllocator* getDeviceAllocator(DeviceType dev_type, uint8_t alloc_flags) {
  return AllocatorFactory::getAllocator(dev_type, alloc_flags);
}

inline DeviceAllocator* getDefaultDeviceAllocator(DeviceType dev_type) {
  return getDeviceAllocator(dev_type, kDefault);
}

}  // namespace ginfer::core::memory
