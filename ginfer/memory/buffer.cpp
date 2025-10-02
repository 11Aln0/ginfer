#include "ginfer/memory/buffer.h"

namespace ginfer::memory {

Buffer::Buffer(size_t size, void* ptr, DeviceType dev_type)
    : size_(size), ptr_(ptr), external_(true), dev_type_(dev_type) {}

Buffer::Buffer(size_t size, std::shared_ptr<DeviceAllocator> allocator)
    : size_(size), external_(false), allocator_(allocator) {
  ptr_ = allocator_->alloc(size_);
  dev_type_ = allocator_->dev_type();
}

Buffer::~Buffer() {
  if (!external_) {
    if (allocator_ && ptr_) {
      allocator_->free(ptr_);
      ptr_ = nullptr;
    }
  }
}

}  // namespace ginfer::memory