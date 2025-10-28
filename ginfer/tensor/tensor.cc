#include <glog/logging.h>
#include <stdexcept>

#include "ginfer/memory/allocator.h"
#include "ginfer/tensor/tensor.h"

namespace ginfer::tensor {

Tensor::Tensor(Dtype dtype, Shape shape, std::shared_ptr<memory::Buffer> buffer)
    : dtype_(dtype), shape_(shape), buffer_(buffer) {
  size_ = shape_.numel();
  if (buffer->size() != size_ * dTypeSize(dtype)) {
    throw std::invalid_argument("Buffer size does not match tensor size.");
  }
}

Tensor::Tensor(Dtype dtype, Shape shape, DeviceType dev_type) : dtype_(dtype), shape_(shape) {
  size_ = shape_.numel();
  buffer_ = std::make_shared<memory::Buffer>(size_ * dTypeSize(dtype), dev_type);
}

const Shape& Tensor::shape() const { return shape_; }

Dtype Tensor::dtype() const { return dtype_; }

size_t Tensor::size() const { return size_; }

size_t Tensor::nbytes() const { return buffer_->size(); }

std::vector<size_t> Tensor::strides() const {
  std::vector<size_t> strides(shape_.ndim(), 1);
  for (int i = shape_.ndim() - 2; i >= 0; --i) {
    strides[i] = strides[i + 1] * shape_[i + 1];
  }
  return strides;
}

void Tensor::toDevice(DeviceType dev_type) {
  CHECK_NE(buffer_, nullptr);
  CHECK_NE(dev_type, DeviceType::kDeviceUnknown);
  CHECK_NE(buffer_->devType(), DeviceType::kDeviceUnknown);
  if (buffer_->devType() != dev_type) {
    auto new_buffer = std::make_shared<memory::Buffer>(buffer_->size(), dev_type);
    new_buffer->copyFrom(buffer_.get());
    this->buffer_ = new_buffer;
  }
}

}  // namespace ginfer::tensor