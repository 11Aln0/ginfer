#include <glog/logging.h>
#include <functional>
#include <numeric>
#include <stdexcept>

#include "ginfer/memory/allocator.h"
#include "ginfer/memory/allocator_factory.h"
#include "ginfer/tensor/tensor.h"

namespace ginfer::tensor {

Tensor::Tensor(DataType dtype, Shape shape, std::shared_ptr<memory::Buffer> buffer, Layout layout)
    : dtype_(dtype), shape_(shape), buffer_(buffer), layout_(layout) {
  size_ = shape_.numel();
  if (buffer->size() != size_ * dTypeSize(dtype)) {
    throw std::invalid_argument("Buffer size does not match tensor size.");
  }
}

Tensor::Tensor(DataType dtype, Shape shape, DeviceType dev_type, Layout layout)
    : dtype_(dtype), shape_(shape), layout_(layout) {
  size_ = shape_.numel();
  buffer_ = std::make_shared<memory::Buffer>(size_ * dTypeSize(dtype), dev_type);
}

Tensor::Tensor(DataType dtype, Shape shape, memory::DeviceAllocator* allocator, Layout layout)
    : dtype_(dtype), shape_(shape), layout_(layout) {
  size_ = shape_.numel();
  buffer_ = std::make_shared<memory::Buffer>(size_ * dTypeSize(dtype), allocator);
}

Layout Tensor::layout() const { return layout_; }

const Shape& Tensor::shape() const { return shape_; }

DataType Tensor::dtype() const { return dtype_; }

size_t Tensor::size() const { return size_; }

size_t Tensor::nbytes() const { return buffer_->size(); }

std::vector<size_t> Tensor::strides() const {
  std::vector<size_t> strides(shape_.ndim(), 1);
  if (layout_ == Layout::kLayoutRowMajor) {
    std::partial_sum(shape_.rbegin(), shape_.rend() - 1, strides.rbegin() + 1, std::multiplies<size_t>());
  } else {
    std::partial_sum(shape_.begin(), shape_.end() - 1, strides.begin() + 1, std::multiplies<size_t>());
  }
  return strides;
}

void Tensor::toDevice(memory::DeviceAllocator* allocator) {
  CHECK_NE(buffer_, nullptr);
  CHECK_NE(allocator, nullptr);
  CHECK_NE(buffer_->devType(), DeviceType::kDeviceUnknown);
  if (buffer_->devType() != allocator->devType()) {
    auto new_buffer = std::make_shared<memory::Buffer>(size_ * dTypeSize(dtype_), allocator);
    new_buffer->copyFrom(buffer_.get());
    this->buffer_ = new_buffer;
  }
}

void Tensor::toDevice(DeviceType dev_type) { toDevice(memory::getDefaultDeviceAllocator(dev_type)); }

}  // namespace ginfer::tensor