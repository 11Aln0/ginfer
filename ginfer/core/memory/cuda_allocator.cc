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
  RETURN_ERR_ON(err != cudaSuccess, "cudaMalloc failed: {}", cudaGetErrorString(err));
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
    const void* src, void* dst, size_t size, MemcpyKind kind, void* stream, bool sync) const {
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

}  // namespace ginfer::core::memory