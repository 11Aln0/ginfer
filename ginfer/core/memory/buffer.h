#pragma once

#include <cstddef>
#include <memory>

#include "ginfer/common/base.h"
#include "ginfer/common/device.h"
#include "ginfer/core/memory/allocator.h"

namespace ginfer::core::memory {

class Buffer : public ginfer::common::NoCopyable, std::enable_shared_from_this<Buffer> {
 public:
  // create buffer from external memory，no need to manage memory release
  static Result<std::shared_ptr<Buffer>, std::string> create(size_t size,
                                                             std::byte* ptr,
                                                             DeviceType dev_type);

  // create buffer by default allocator，need to manage memory release
  static Result<std::shared_ptr<Buffer>, std::string> create(size_t size, DeviceType dev_type);

  // create buffer by allocator, need to manage memory release
  static Result<std::shared_ptr<Buffer>, std::string> create(size_t size,
                                                             DeviceAllocator* allocator);

 public:
  void copyFrom(const std::shared_ptr<Buffer>& src, bool async = false);
  void copyFrom(const Buffer& src, bool async = false);

  void copyFrom(const std::shared_ptr<Buffer>& src,
                int64_t src_off,
                int64_t dst_off,
                size_t size,
                bool async = false);
  void copyFrom(const Buffer& src,
                int64_t src_off,
                int64_t dst_off,
                size_t size,
                bool async = false);

  virtual ~Buffer();

  size_t size() const { return size_; }

  DeviceType devType() const { return dev_type_; }

  DeviceAllocator* allocator() const { return allocator_; }

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

}  // namespace ginfer::core::memory