#pragma once

#include <glog/logging.h>
#include "ginfer/memory/allocator.h"
#include "ginfer/memory/cuda/alloc_strategy.h"

namespace ginfer::memory::cuda {

template <typename S>
class CUDADeviceAllocator : public DeviceAllocator {
 public:
  explicit CUDADeviceAllocator() : DeviceAllocator(DeviceType::kDeviceCUDA) {}

  void* alloc(size_t size) override { return strategy_.alloc(size); }

  void free(void* ptr, size_t size) override { strategy_.free(ptr, size); }

  void memcpy(const void* src, void* dst, size_t size, MemcpyKind kind, void* stream = nullptr, bool sync = false) const override {
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

 private:
  S strategy_;
};

}  // namespace ginfer::memory::cuda