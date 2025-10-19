#pragma once

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <type_traits>

#include "ginfer/common/device.h"

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

  virtual void* alloc(size_t size) const = 0;

  virtual void free(void* ptr) const = 0;

  virtual void memcpy(const void* src, void* dst, size_t size, MemcpyKind kind,
                      void* stream = nullptr, bool sync = false) const = 0;

  // virtual void memset(void* ptr, size_t size, char c) {}

 private:
  DeviceType dev_type_;
};

class CPUDeviceAllocator : public DeviceAllocator {
 public:
  explicit CPUDeviceAllocator();

  void* alloc(size_t size) const override;

  void free(void* ptr) const override;

  void memcpy(const void* src, void* dst, size_t size, MemcpyKind kind, void* stream = nullptr,
              bool sync = false) const override;
};

class CUDADeviceAllocator : public DeviceAllocator {
 public:
  explicit CUDADeviceAllocator();

  void* alloc(size_t size) const override;

  void free(void* ptr) const override;

  void memcpy(const void* src, void* dst, size_t size, MemcpyKind kind, void* stream = nullptr,
              bool sync = false) const override;
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

std::shared_ptr<DeviceAllocator> getDeviceAllocator(DeviceType dev_type);

}  // namespace ginfer::memory
