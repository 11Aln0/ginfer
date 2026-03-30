#pragma once

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>

#include "ginfer/common/device.h"
#include "ginfer/common/errors.h"
#include "ginfer/core/memory/alloc_stats.h"

namespace ginfer::core::memory {

using DeviceType = ginfer::common::DeviceType;

enum class MemcpyKind {
  kMemcpyHostToHost,
  kMemcpyHostToDevice,
  kMemcpyDeviceToHost,
  kMemcpyDeviceToDevice,
};

struct DeviceMemInfo {
  size_t total;  // bytes
  size_t free;
};

class DeviceAllocator : protected AllocatorStatsTracker {
 public:
  explicit DeviceAllocator(DeviceType dev_type) : dev_type_(dev_type){};

  DeviceType devType() const { return dev_type_; }

  Result<void*, std::string> alloc(size_t size);

  // free should not have size argument, but like pooled allocator we need size to find the
  // right pool in order not to maintain a map from ptr to size (introduce searching and locking
  // overhead), we add size argument here might have a better way to implement it in future
  void free(void* ptr, size_t size);

  void setStream(void* stream);

  virtual void memcpy(
      const void* src, void* dst, size_t size, MemcpyKind kind, bool async = false) const = 0;

  virtual DeviceMemInfo getMemInfo() const = 0;

  virtual ~DeviceAllocator() = default;

  using AllocatorStatsTracker::getStats;
  using AllocatorStatsTracker::reset;

 protected:
  virtual Result<void*, std::string> doAlloc(size_t size) = 0;

  virtual void doFree(void* ptr, size_t size) = 0;

 protected:
  void* stream_ = nullptr;  // optional stream for async copy, only used by some devices like CUDA

 private:
  DeviceType dev_type_;
};

class CPUDeviceAllocator : public DeviceAllocator {
 public:
  explicit CPUDeviceAllocator();

  void memcpy(
      const void* src, void* dst, size_t size, MemcpyKind kind, bool async = false) const override;

  DeviceMemInfo getMemInfo() const override;

 protected:
  Result<void*, std::string> doAlloc(size_t size) override;

  void doFree(void* ptr, size_t size) override;
};

class PinnedCPUAllocator : public DeviceAllocator {
 public:
  explicit PinnedCPUAllocator();

  void memcpy(
      const void* src, void* dst, size_t size, MemcpyKind kind, bool async = false) const override;

  DeviceMemInfo getMemInfo() const override;

 protected:
  Result<void*, std::string> doAlloc(size_t size) override;

  void doFree(void* ptr, size_t size) override;
};

class CUDADeviceAllocator : public DeviceAllocator {
 public:
  explicit CUDADeviceAllocator();

  void memcpy(
      const void* src, void* dst, size_t size, MemcpyKind kind, bool async = false) const override;

  DeviceMemInfo getMemInfo() const override;

 protected:
  Result<void*, std::string> doAlloc(size_t size) override;
  void doFree(void* ptr, size_t size) override;
};

}  // namespace ginfer::core::memory
