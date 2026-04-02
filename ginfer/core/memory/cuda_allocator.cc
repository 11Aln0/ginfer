#include <cuda_runtime.h>
#include <glog/logging.h>
#include "ginfer/core/memory/allocator.h"

namespace ginfer::core::memory {

CUDADeviceAllocator::CUDADeviceAllocator() : DeviceAllocator(DeviceType::kDeviceCUDA) {}

Result<void*, std::string> CUDADeviceAllocator::doAlloc(size_t size) {
  if (size == 0) {
    LOG(WARNING) << "Try to allocate 0 bytes.";
    return Ok((void*)nullptr);
  }
  void* ptr = nullptr;
  cudaError_t err = cudaMalloc(&ptr, size);
  RETURN_ERR_ON(err != cudaSuccess, "trying to allocate {} bytes by cudaMalloc, reason: {}", size,
                cudaGetErrorString(err));
  DLOG(INFO) << "Allocated " << size / 1024.0 << " KB on CUDA device.";
  return Ok(ptr);
}

void CUDADeviceAllocator::doFree(void* ptr, size_t size) {
  (void)size;
  if (!ptr) {
    LOG(WARNING) << "Try to free a nullptr.";
    return;
  }
  cudaError_t err = cudaFree(ptr);
  CHECK_EQ(err, cudaSuccess) << "cudaFree failed: "
                             << cudaGetErrorString(err);  // should not happen
}

void CUDADeviceAllocator::memcpy(
    const void* src, void* dst, size_t size, MemcpyKind kind, bool async) const {
  CHECK_NE(src, nullptr);
  CHECK_NE(dst, nullptr);
  if (size == 0) {
    LOG(WARNING) << "Try to copy 0 bytes.";
    return;
  }

  cudaMemcpyKind cu_kind;

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

  cudaError_t err;
  if (async) {
    cudaStream_t stream = stream_ ? static_cast<cudaStream_t>(stream_) : cudaStreamDefault;
    err = cudaMemcpyAsync(dst, src, size, cu_kind, stream);
  } else {
    err = cudaMemcpy(dst, src, size, cu_kind);
  }
  CHECK(err == cudaSuccess) << "cudaMemcpy failed: " << cudaGetErrorString(err);
}

DeviceMemInfo CUDADeviceAllocator::getMemInfo() const {
  size_t free_bytes = 0, total_bytes = 0;
  cudaError_t err = cudaMemGetInfo(&free_bytes, &total_bytes);
  CHECK_EQ(err, cudaSuccess) << "cudaMemGetInfo failed: " << cudaGetErrorString(err);
  return {total_bytes, free_bytes};
}

}  // namespace ginfer::core::memory