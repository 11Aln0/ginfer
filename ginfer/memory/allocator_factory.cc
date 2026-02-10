#include "ginfer/memory/allocator_factory.h"
#include "ginfer/memory/cuda/alloc_strategy.h"

namespace ginfer::memory {

DeviceAllocator* getDefaultDeviceAllocator(DeviceType dev_type) {
  switch (dev_type) {
    case DeviceType::kDeviceCPU:
      return GlobalCPUAllocator::getInstance();
    case DeviceType::kDeviceCUDA:
      return GlobalCUDAAllocator<cuda::DefaultAllocStrategy>::getInstance();
    default:
      throw std::invalid_argument("Unsupported device type for allocator.");
  }
}

}  // namespace ginfer::memory