#pragma once

#include <cstddef>
#include <memory>

#include "ginfer/common/base.h"
#include "ginfer/common/device.h"
#include "ginfer/memory/allocator.h"

namespace ginfer::memory {

class Buffer : public ginfer::common::NoCopyable, std::enable_shared_from_this<Buffer> {
 public:
  // create buffer from external memory，no need to manage memory release
  static Result<std::shared_ptr<Buffer>, std::string> create(size_t size, std::byte* ptr, DeviceType dev_type);

  // create buffer by default allocator，need to manage memory release
  static Result<std::shared_ptr<Buffer>, std::string> create(size_t size, DeviceType dev_type);

  // create buffer by allocator, need to manage memory release
  static Result<std::shared_ptr<Buffer>, std::string> create(size_t size, DeviceAllocator* allocator);

 public:
  void copyFrom(const Buffer& src);

  void copyFrom(const Buffer* src);

  void copyFrom(const Buffer& src, size_t size);

  void copyFrom(const Buffer* src, size_t size);

  virtual ~Buffer();

  size_t size() const { return size_; }

  DeviceType devType() const { return dev_type_; }

  std::byte* ptr() const { return ptr_; }

  bool allocated() const { return ptr_ != nullptr; }

 private:
  explicit Buffer() = delete;

  explicit Buffer(size_t size, std::byte* ptr, DeviceAllocator* allocator, bool external);

 private:
  size_t size_ = 0;
  std::byte* ptr_ = nullptr;
  bool external_ = false;
  DeviceType dev_type_ = DeviceType::kDeviceUnknown;
  DeviceAllocator* allocator_ = nullptr;
};

}  // namespace ginfer::memory