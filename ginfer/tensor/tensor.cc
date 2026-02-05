#include <fmt/core.h>
#include <glog/logging.h>
#include <functional>
#include <numeric>
#include <stdexcept>

#include "ginfer/memory/allocator.h"
#include "ginfer/memory/allocator_factory.h"
#include "ginfer/tensor/tensor.h"

namespace ginfer::tensor {

Tensor::Tensor(DataType dtype, Shape shape, std::shared_ptr<memory::Buffer> buffer, Layout layout)
    : dtype_(dtype), shape_(shape), buffer_(buffer), layout_(layout), offset_(0) {
  size_ = shape_.numel();
  if (buffer->size() < size_ * dTypeSize(dtype)) {
    throw std::invalid_argument(
        fmt::format("Buffer size is smaller than tensor size. Buffer size: {}, required size: {}", buffer->size(),
                    size_ * dTypeSize(dtype)));
  }
  calcStrides();
}

Tensor::Tensor(DataType dtype, Shape shape, DeviceType dev_type, Layout layout)
    : Tensor(dtype, shape, std::make_shared<memory::Buffer>(shape.numel() * dTypeSize(dtype), dev_type), layout) {}

Tensor::Tensor(DataType dtype, Shape shape, memory::DeviceAllocator* allocator, Layout layout)
    : Tensor(dtype, shape, std::make_shared<memory::Buffer>(shape.numel() * dTypeSize(dtype), allocator), layout) {}

Layout Tensor::layout() const { return layout_; }

const Shape& Tensor::shape() const { return shape_; }

DataType Tensor::dtype() const { return dtype_; }

size_t Tensor::size() const { return size_; }

size_t Tensor::nbytes() const { return buffer_->size(); }

std::vector<ptrdiff_t> Tensor::strides() const { return strides_; }

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

std::shared_ptr<Tensor> Tensor::slice(int dim, int64_t start, int64_t end) const {
  CHECK(dim >= 0 && dim < static_cast<int>(shape_.ndim())) << "Slice failed: dimension out of range.";
  CHECK(start >= 0 && end <= shape_[dim] && start < end) << "Slice failed: invalid start or end indices.";

  Shape new_shape = shape_;
  new_shape[dim] = end - start;
  // TODO more tidy way to implement this
  auto new_tensor = std::make_shared<Tensor>(dtype_, new_shape, buffer_, layout_);
  size_t new_offset = start * strides_[dim] + offset_;
  new_tensor->offset_ = new_offset;
  new_tensor->strides_ = strides_;
  return new_tensor;
}

std::shared_ptr<Tensor> Tensor::reshape(const Shape& new_shape) const {
  CHECK_EQ(shape_.numel(), new_shape.numel()) << "Reshape failed: number of elements does not match.";
  auto new_tensor = std::make_shared<Tensor>(dtype_, new_shape, buffer_, layout_);
  new_tensor->offset_ = offset_;
  return new_tensor;
}

void Tensor::calcStrides() {
  size_t ndim = shape_.ndim();
  strides_.resize(ndim);
  if (layout_ == Layout::kLayoutRowMajor) {
    strides_[ndim - 1] = 1;
    std::partial_sum(shape_.rbegin(), shape_.rend() - 1, strides_.rbegin() + 1, std::multiplies<ptrdiff_t>());
  } else {
    strides_[0] = 1;
    std::partial_sum(shape_.begin(), shape_.end() - 1, strides_.begin() + 1, std::multiplies<ptrdiff_t>());
  }
}

}  // namespace ginfer::tensor