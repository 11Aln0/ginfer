#include "ginfer/alloc/alloctor.h"
#include <cstdlib>
#include <cstring>

CPUDeviceAllocator::CPUDeviceAllocator(): DeviceAllocator(DeviceType::kDeviceCPU) {}

void* CPUDeviceAllocator::alloc(size_t size) const {
    return malloc(size);
}

void CPUDeviceAllocator::free(void* ptr) const {
    free(ptr);
}

void CPUDeviceAllocator::memcpy(const void* src, void* dst, size_t size, MemcpyKind kind) const {
    (void)kind;

    std::memcpy(dst, src, size);
}


