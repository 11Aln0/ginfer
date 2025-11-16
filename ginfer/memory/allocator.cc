#include "allocator.h"
#include <cstring>

namespace ginfer::memory {

CPUDeviceAllocator::CPUDeviceAllocator() : DeviceAllocator(DeviceType::kDeviceCPU) {}

void* CPUDeviceAllocator::alloc(size_t size) { return malloc(size); }

void CPUDeviceAllocator::free(void* ptr, size_t size) {
  (void)size;
  std::free(ptr);
}

void CPUDeviceAllocator::memcpy(const void* src, void* dst, size_t size, MemcpyKind kind, void* stream, bool sync) const {
  (void)kind;
  (void)stream;
  (void)sync;
  std::memcpy(dst, src, size);
}

}  // namespace ginfer::memory