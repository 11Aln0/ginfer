#include "ginfer/memory/buffer.h"
#include <glog/logging.h>
#include "ginfer/memory/allocator_factory.h"

namespace ginfer::memory {

Result<std::shared_ptr<Buffer>, std::string> Buffer::create(size_t size, std::byte* ptr, DeviceType dev_type) {
  auto allocator = getDefaultDeviceAllocator(dev_type);
  return Ok(std::shared_ptr<Buffer>(new Buffer(size, ptr, allocator, true)));
}

Result<std::shared_ptr<Buffer>, std::string> Buffer::create(size_t size, DeviceAllocator* allocator) {
  DECLARE_OR_RETURN(ptr, allocator->alloc(size));
  return Ok(std::shared_ptr<Buffer>(new Buffer(size, (std::byte*)ptr, allocator, false)));
}

Result<std::shared_ptr<Buffer>, std::string> Buffer::create(size_t size, DeviceType dev_type) {
  return create(size, getDefaultDeviceAllocator(dev_type));
}

Buffer::Buffer(size_t size, std::byte* ptr, DeviceAllocator* allocator, bool external)
    : size_(size), ptr_(ptr), external_(external), dev_type_(allocator->devType()), allocator_(allocator) {}

void Buffer::copyFrom(const Buffer& src, size_t size) {
  CHECK(size_ <= size && size <= src.size())
      << "Size argument is out of range. Size: " << size << ", this buffer size: " << size_
      << ", source buffer size: " << src.size();
  CHECK(src.ptr() != nullptr && ptr_ != nullptr) << "Source or destination buffer pointer is nullptr.";

  if (src.devType() == dev_type_) {
    allocator_->memcpy(src.ptr(), ptr_, size_, MemcpyKind::kMemcpyDeviceToDevice);
  } else {
    if (common::isHostDevice(src.devType())) {
      allocator_->memcpy(src.ptr(), ptr_, size_, MemcpyKind::kMemcpyHostToDevice);  // src is host
    } else {
      src.allocator_->memcpy(src.ptr(), ptr_, size_,
                             MemcpyKind::kMemcpyDeviceToHost);  // src is device
    }
  }
}

void Buffer::copyFrom(const Buffer* src, size_t size) {
  CHECK(src != nullptr) << "Source buffer is nullptr.";
  copyFrom(*src, size);
}

void Buffer::copyFrom(const Buffer& src) { copyFrom(src, size_); }

void Buffer::copyFrom(const Buffer* src) { copyFrom(*src, size_); }

Buffer::~Buffer() {
  if (!external_) {
    allocator_->free(ptr_, size_);
    ptr_ = nullptr;
  }
}

}  // namespace ginfer::memory