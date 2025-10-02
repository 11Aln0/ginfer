#pragma once

#include <memory>

#include "ginfer/memory/alloctor.h"
#include "ginfer/memory/base.h"
#include "ginfer/memory/device.h"

namespace ginfer::memory {

class Buffer : public NoCopyable, std::enable_shared_from_this<Buffer> {
 private:
  size_t size_ = 0;
  void* ptr_ = nullptr;
  bool external_ = false;
  DeviceType dev_type_ = DeviceType::kDeviceUnknown;
  std::shared_ptr<DeviceAllocator> allocator_ = nullptr;

 public:
  explicit Buffer() = default;

  explicit Buffer(size_t size, void* ptr, DeviceType dev_type);

  explicit Buffer(size_t size, std::shared_ptr<DeviceAllocator> allocator);

  virtual ~Buffer();

  size_t size() const { return size_; }

  DeviceType dev_type() const { return dev_type_; }

  void* ptr() const { return ptr_; }
};

}  // namespace ginfer::memory