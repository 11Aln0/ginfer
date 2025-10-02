#include <cstdlib>
#include <cstring>

#include "ginfer/memory/alloctor.h"

namespace ginfer::memory {

CPUDeviceAllocator::CPUDeviceAllocator() : DeviceAllocator(DeviceType::kDeviceCPU) {}

void* CPUDeviceAllocator::alloc(size_t size) const { return malloc(size); }

void CPUDeviceAllocator::free(void* ptr) const { std::free(ptr); }

void CPUDeviceAllocator::memcpy(const void* src, void* dst, size_t size, MemcpyKind kind,
                                void* stream, bool sync) const {
  (void)kind;
  (void)stream;
  (void)sync;
  std::memcpy(dst, src, size);
}

// using CPUAllocatorFactory = DeviceAllocatorFactory<CPUDeviceAllocator>;
// auto CPUAllocatorFactory::instance = nullptr;

}  // namespace ginfer::memory