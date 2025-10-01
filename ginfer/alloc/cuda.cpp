#include "ginfer/alloc/alloctor.h"
#include <cuda_runtime_api.h>

CUDADeviceAllocator::CUDADeviceAllocator(): DeviceAllocator(DeviceType::kDeviceCUDA) {}

void* CUDADeviceAllocator::alloc(size_t size) const {
    return nullptr;
}