#include "ginfer/tensor/tensor.h"

#include <stdexcept>

namespace ginfer::tensor {

Tensor::Tensor(DType dtype, Shape shape, std::shared_ptr<memory::Buffer> buffer)
    : dtype_(dtype), shape_(shape), buffer_(buffer) {
  size_ = shape_.numel();
  if (buffer->size() != size_ * dTypeSize(dtype)) {
    throw std::invalid_argument("Buffer size does not match tensor size.");
  }
}

Tensor::Tensor(DType dtype, Shape shape, std::shared_ptr<memory::DeviceAllocator> allocator)
    : dtype_(dtype), shape_(shape) {
  size_ = shape_.numel();
  buffer_ = std::make_shared<memory::Buffer>(size_ * dTypeSize(dtype), allocator);
}

const Shape& Tensor::shape() const { return shape_; }

DType Tensor::dtype() const { return dtype_; }

size_t Tensor::size() const { return size_; }

size_t Tensor::nbytes() const { return buffer_->size(); }

std::vector<size_t> Tensor::strides() const {
  std::vector<size_t> strides(shape_.ndim(), 1);
  for (int i = shape_.ndim() - 2; i >= 0; --i) {
    strides[i] = strides[i + 1] * shape_[i + 1];
  }
  return strides;
}

}  // namespace ginfer::tensor