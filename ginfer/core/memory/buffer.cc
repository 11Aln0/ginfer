#include "ginfer/core/memory/buffer.h"
#include <glog/logging.h>
#include "ginfer/core/memory/allocator_factory.h"

namespace ginfer::core::memory {

Result<std::shared_ptr<Buffer>, std::string> Buffer::create(size_t size,
                                                            std::byte* ptr,
                                                            DeviceType dev_type) {
  auto allocator = getDefaultDeviceAllocator(dev_type);
  return Ok(std::shared_ptr<Buffer>(new Buffer(size, ptr, allocator, true)));
}

Result<std::shared_ptr<Buffer>, std::string> Buffer::create(size_t size,
                                                            DeviceAllocator* allocator) {
  DECLARE_OR_RETURN(ptr, allocator->alloc(size));
  return Ok(std::shared_ptr<Buffer>(new Buffer(size, (std::byte*)ptr, allocator, false)));
}

Result<std::shared_ptr<Buffer>, std::string> Buffer::create(size_t size, DeviceType dev_type) {
  return create(size, getDefaultDeviceAllocator(dev_type));
}

Buffer::Buffer(size_t size, std::byte* ptr, DeviceAllocator* allocator, bool external)
    : size_(size),
      ptr_(ptr),
      external_(external),
      dev_type_(allocator->devType()),
      allocator_(allocator) {}

void Buffer::copyFrom(const Buffer& src, int64_t src_off, int64_t dst_off, size_t size) {
  CHECK(src_off >= 0 && dst_off >= 0) << "Copy failed: src/dst offset cannot be negative.";
  CHECK(src.size() >= src_off + size)
      << "Copy failed: source buffer size is smaller than copy size.";
  CHECK(size_ >= dst_off + size)
      << "Copy failed: destination buffer size is smaller than copy size.";

  if (src.devType() == dev_type_) {
    allocator_->memcpy(src.ptr() + src_off, ptr_ + dst_off, size,
                       MemcpyKind::kMemcpyDeviceToDevice);
  } else {
    if (common::isHostDevice(src.devType())) {
      allocator_->memcpy(src.ptr() + src_off, ptr_ + dst_off, size,
                         MemcpyKind::kMemcpyHostToDevice);  // src is host
    } else {
      src.allocator_->memcpy(src.ptr() + src_off, ptr_ + dst_off, size,
                             MemcpyKind::kMemcpyDeviceToHost);  // src is device
    }
  }
}

void Buffer::copyFrom(const std::shared_ptr<Buffer>& src,
                      int64_t src_off,
                      int64_t dst_off,
                      size_t size) {
  CHECK(src != nullptr) << "Source buffer is nullptr.";
  copyFrom(*src, src_off, dst_off, size);
}

void Buffer::copyFrom(const std::shared_ptr<Buffer>& src) { copyFrom(src, 0, 0, size_); }

void Buffer::copyFrom(const Buffer& src) { copyFrom(src, 0, 0, size_); }

Buffer::~Buffer() {
  if (!external_) {
    allocator_->free(ptr_, size_);
    ptr_ = nullptr;
  }
}

}  // namespace ginfer::core::memory