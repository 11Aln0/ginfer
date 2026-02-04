#include "ginfer/memory/buffer.h"
#include <glog/logging.h>
#include "ginfer/memory/allocator_factory.h"

namespace ginfer::memory {

Buffer::Buffer(size_t size, std::byte* ptr, DeviceType dev_type)
    : size_(size), ptr_(ptr), external_(true), dev_type_(dev_type) {
  allocator_ = getDefaultDeviceAllocator(dev_type);
}

Buffer::Buffer(size_t size, DeviceType dev_type) : size_(size), external_(false), dev_type_(dev_type) {
  allocator_ = getDefaultDeviceAllocator(dev_type);
  ptr_ = (std::byte*)allocator_->alloc(size_);
}

Buffer::Buffer(size_t size, DeviceAllocator* allocator)
    : size_(size), external_(false), dev_type_(allocator->devType()), allocator_(allocator) {
  ptr_ = (std::byte*)allocator_->alloc(size_);
}

void Buffer::copyFrom(const Buffer& src) {
  CHECK(src.size() == size_) << "Source buffer size (" << src.size() << ") does not match destination buffer size ("
                             << size_ << ").";
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

void Buffer::copyFrom(const Buffer* src) {
  CHECK(src != nullptr) << "Source buffer is nullptr.";
  copyFrom(*src);
}

Buffer::~Buffer() {
  if (!external_) {
    allocator_->free(ptr_, size_);
    ptr_ = nullptr;
  }
}

}  // namespace ginfer::memory