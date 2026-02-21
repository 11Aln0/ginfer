#pragma once

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>

#include "ginfer/common/device.h"
#include "ginfer/common/errors.h"

namespace ginfer::memory {

using DeviceType = ginfer::common::DeviceType;

enum class MemcpyKind {
  kMemcpyHostToHost,
  kMemcpyHostToDevice,
  kMemcpyDeviceToHost,
  kMemcpyDeviceToDevice,
};

class DeviceAllocator {
 public:
  explicit DeviceAllocator(DeviceType dev_type) : dev_type_(dev_type){};

  DeviceType devType() const { return dev_type_; }

  virtual Result<void*, std::string> alloc(size_t size) = 0;

  // free should not have size argument, but like pooled allocator we need size to find the right pool
  // in order not to maintain a map from ptr to size (introduce searching and locking overhead), we add size argument
  // here might have a better way to implement it in future
  virtual void free(void* ptr, size_t size) = 0;

  virtual void memcpy(const void* src, void* dst, size_t size, MemcpyKind kind, void* stream = nullptr,
                      bool sync = false) const = 0;

  // virtual void memset(void* ptr, size_t size, char c) {}

 private:
  DeviceType dev_type_;
};

class CPUDeviceAllocator : public DeviceAllocator {
 public:
  explicit CPUDeviceAllocator();

  Result<void*, std::string> alloc(size_t size) override;

  void free(void* ptr, size_t size) override;

  void memcpy(const void* src, void* dst, size_t size, MemcpyKind kind, void* stream = nullptr,
              bool sync = false) const override;
};

class CUDADeviceAllocator : public DeviceAllocator {
 public:
  explicit CUDADeviceAllocator();

  Result<void*, std::string> alloc(size_t size) override;

  void free(void* ptr, size_t size) override;

  void memcpy(const void* src, void* dst, size_t size, MemcpyKind kind, void* stream = nullptr,
              bool sync = false) const override;
};

}  // namespace ginfer::memory
