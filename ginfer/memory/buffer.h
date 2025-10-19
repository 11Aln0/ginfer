#pragma once

#include <memory>

#include "ginfer/common/base.h"
#include "ginfer/common/device.h"
#include "ginfer/memory/allocator.h"

namespace ginfer::memory {

class Buffer : public ginfer::common::NoCopyable, std::enable_shared_from_this<Buffer> {
 private:
  size_t size_ = 0;
  void* ptr_ = nullptr;
  bool external_ = false;
  DeviceType dev_type_ = DeviceType::kDeviceUnknown;
  std::shared_ptr<DeviceAllocator> allocator_ = nullptr;

 public:
  explicit Buffer() = default;

  // create buffer from external memory，no need to manage memory release
  explicit Buffer(size_t size, void* ptr, DeviceType dev_type);

  // create buffer from allocator，need to manage memory release
  explicit Buffer(size_t size, DeviceType dev_type);

  void copyFrom(const Buffer& src);

  void copyFrom(const Buffer* src);

  virtual ~Buffer();

  size_t size() const { return size_; }

  DeviceType devType() const { return dev_type_; }

  void* ptr() const { return ptr_; }
};

}  // namespace ginfer::memory