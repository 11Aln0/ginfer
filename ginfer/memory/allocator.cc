#include "allocator.h"

namespace ginfer::memory {

std::shared_ptr<DeviceAllocator> getDeviceAllocator(DeviceType dev_type) {
  switch (dev_type) {
    case DeviceType::kDeviceCPU:
      return CPUAllocatorFactory::get_instance();
    case DeviceType::kDeviceCUDA:
      return CUDAAllocatorFactory::get_instance();
    default:
      throw std::invalid_argument("Unsupported device type for allocator.");
  }
}

}  // namespace ginfer::memory