#include <cuda_runtime.h>
#include <glog/logging.h>
#include <sys/sysinfo.h>

#include "ginfer/core/memory/allocator.h"
#include "ginfer/common/errors.h"

namespace ginfer::core::memory {

PinnedCPUAllocator::PinnedCPUAllocator() : DeviceAllocator(DeviceType::kDeviceCPU) {}

Result<void*, std::string> PinnedCPUAllocator::doAlloc(size_t size) {
  if (size == 0) {
    LOG(WARNING) << "Try to allocate 0 bytes.";
    return Ok((void*)nullptr);
  }
  void* ptr = nullptr;
  cudaError_t err = cudaMallocHost(&ptr, size);
  RETURN_ERR_ON(err != cudaSuccess, "cudaMallocHost failed: {}", cudaGetErrorString(err));
  return Ok(ptr);
}

void PinnedCPUAllocator::doFree(void* ptr, size_t size) {
  (void)size;
  if (!ptr) {
    LOG(WARNING) << "Try to free a nullptr.";
    return;
  }
  cudaError_t err = cudaFreeHost(ptr);
  CHECK_EQ(err, cudaSuccess) << "cudaFreeHost failed: " << cudaGetErrorString(err);
}

void PinnedCPUAllocator::memcpy(
    const void* src, void* dst, size_t size, MemcpyKind kind, bool async) const {
  (void)async;
  CHECK_NE(src, nullptr);
  CHECK_NE(dst, nullptr);
  CHECK(kind == MemcpyKind::kMemcpyHostToHost)
      << "PinnedCPUAllocator only supports host-to-host copies.";
  if (size == 0) {
    LOG(WARNING) << "Try to copy 0 bytes.";
    return;
  }
  cudaError_t err = cudaMemcpy(dst, src, size, cudaMemcpyHostToHost);
  CHECK_EQ(err, cudaSuccess) << "cudaMemcpy(HostToHost) failed: " << cudaGetErrorString(err);
}

DeviceMemInfo PinnedCPUAllocator::getMemInfo() const {
  struct sysinfo info;
  if (sysinfo(&info) == 0) {
    return {info.totalram * info.mem_unit, info.freeram * info.mem_unit};
  }
  return {0, 0};
}

}  // namespace ginfer::core::memory
