#pragma once

#include <cstddef>
#include <memory>
#include <type_traits>

#include "ginfer/memory/device.h"

namespace ginfer::memory {

enum class MemcpyKind { kMemcpyHostToDevice, kMemcpyDeviceToHost };

class DeviceAllocator {
 public:
  explicit DeviceAllocator(DeviceType dev_type) : dev_type_(dev_type){};

  DeviceType devType() const { return dev_type_; }

  virtual void* alloc(size_t size) const = 0;

  virtual void free(void* ptr) const = 0;

  virtual void memcpy(const void* src, void* dst, size_t size, MemcpyKind kind, void* stream,
                      bool sync) const = 0;

  // virtual void memset(void* ptr, size_t size, char c) {}

 private:
  DeviceType dev_type_;
};

class CPUDeviceAllocator : public DeviceAllocator {
 public:
  explicit CPUDeviceAllocator();

  void* alloc(size_t size) const override;

  void free(void* ptr) const override;

  void memcpy(const void* src, void* dst, size_t size, MemcpyKind kind, void* stream,
              bool sync) const override;
};

class CUDADeviceAllocator : public DeviceAllocator {
 public:
  explicit CUDADeviceAllocator();

  void* alloc(size_t size) const override;

  void free(void* ptr) const override;

  void memcpy(const void* src, void* dst, size_t size, MemcpyKind kind, void* stream,
              bool sync) const override;
};

template <class T, typename = typename std::enable_if_t<std::is_base_of<DeviceAllocator, T>::value>>
class DeviceAllocatorFactory {
 public:
  static std::shared_ptr<T> get_instance() {
    if (!instance) {
      instance = std::make_shared<T>();
    }
    return instance;
  }

 private:
  inline static std::shared_ptr<T> instance = nullptr;
};

using CUDAAllocatorFactory = DeviceAllocatorFactory<CUDADeviceAllocator>;
using CPUAllocatorFactory = DeviceAllocatorFactory<CPUDeviceAllocator>;

}  // namespace ginfer::memory
