#include <cuda_runtime_api.h>
#include <glog/logging.h>

#include "ginfer/memory/allocator.h"

namespace ginfer::memory {
CUDADeviceAllocator::CUDADeviceAllocator() : DeviceAllocator(DeviceType::kDeviceCUDA) {}

void* CUDADeviceAllocator::alloc(size_t size) const {
  if (size == 0) {
    LOG(WARNING) << "Try to allocate 0 bytes.";
    return nullptr;
  }
  void* ptr = nullptr;
  cudaError_t err = cudaMalloc(&ptr, size);
  CHECK_EQ(err, cudaSuccess) << "cudaMalloc failed: " << cudaGetErrorString(err);
  return ptr;
}

void CUDADeviceAllocator::free(void* ptr) const {
  if (!ptr) {
    LOG(WARNING) << "Try to free a nullptr.";
    return;
  }
  cudaError_t err = cudaFree(ptr);
  CHECK_EQ(err, cudaSuccess) << "cudaFree failed: " << cudaGetErrorString(err);
}

void CUDADeviceAllocator::memcpy(const void* src, void* dst, size_t size, MemcpyKind kind,
                                 void* stream, bool sync) const {
  CHECK_NE(src, nullptr);
  CHECK_NE(dst, nullptr);
  if (size == 0) {
    LOG(WARNING) << "Try to copy 0 bytes.";
    return;
  }

  cudaStream_t cu_stream = nullptr;
  if (stream) cu_stream = static_cast<cudaStream_t>(stream);

  cudaMemcpyKind cu_kind = cudaMemcpyDefault;

  switch (kind) {
    case MemcpyKind::kMemcpyHostToDevice:
      cu_kind = cudaMemcpyHostToDevice;
      break;
    case MemcpyKind::kMemcpyDeviceToHost:
      cu_kind = cudaMemcpyDeviceToHost;
      break;
    case MemcpyKind::kMemcpyDeviceToDevice:
      cu_kind = cudaMemcpyDeviceToDevice;
      break;
    default:
      LOG(FATAL) << "Unsupported MemcpyKind.";
  }

  if (cu_stream) {
    cudaMemcpyAsync(dst, src, size, cu_kind, cu_stream);
  } else {
    cudaMemcpy(dst, src, size, cu_kind);
  }

  if (sync) {
    cudaDeviceSynchronize();
  }
}

// using CUDAAllocatorFactory = DeviceAllocatorFactory<CUDADeviceAllocator>;
// auto CUDAAllocatorFactory::instance = nullptr;

}  // namespace ginfer::memory
