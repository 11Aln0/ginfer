#include "ginfer/memory/allocator_factory.h"
#include "ginfer/memory/cuda/alloc_strategy.h"

namespace ginfer::memory {

DeviceAllocator* getDefaultDeviceAllocator(DeviceType dev_type) {
  switch (dev_type) {
    case DeviceType::kDeviceCPU:
      return CPUAllocatorFactory::get_instance();
    case DeviceType::kDeviceCUDA:
      return CUDAAllocatorFactory<cuda::DefaultAllocStrategy>::get_instance();
    default:
      throw std::invalid_argument("Unsupported device type for allocator.");
  }
}

}  // namespace ginfer::memory