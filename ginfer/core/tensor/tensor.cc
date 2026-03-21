#include <fmt/core.h>
#include <glog/logging.h>
#include <functional>
#include <numeric>
#include <stdexcept>

#include "ginfer/core/memory/allocator.h"
#include "ginfer/core/memory/allocator_factory.h"
#include "ginfer/core/tensor/tensor.h"

namespace ginfer::core::tensor {

Result<TensorRef, std::string> Tensor::create(DataType dtype,
                                              Shape shape,
                                              std::shared_ptr<memory::Buffer> buffer) {
  return Ok(std::shared_ptr<Tensor>(new Tensor(dtype, shape, buffer)));
}

Result<TensorRef, std::string> Tensor::create(DataType dtype, Shape shape, DeviceType dev_type) {
  DECLARE_OR_RETURN(buffer, memory::Buffer::create(shape.numel() * dTypeSize(dtype), dev_type));
  return create(dtype, std::move(shape), buffer);
}

Result<TensorRef, std::string> Tensor::create(DataType dtype,
                                              Shape shape,
                                              DeviceAllocator* allocator) {
  DECLARE_OR_RETURN(buffer, memory::Buffer::create(shape.numel() * dTypeSize(dtype), allocator));
  return create(dtype, std::move(shape), buffer);
}

Tensor::Tensor(DataType dtype, Shape shape, std::shared_ptr<memory::Buffer> buffer)
    : dtype_(dtype), shape_(shape), buffer_(buffer), offset_(0) {
  size_ = shape_.numel();
  if (buffer->size() < size_ * dTypeSize(dtype)) {
    throw std::invalid_argument(
        fmt::format("Buffer size is smaller than tensor size. Buffer size: {}, required size: {}",
                    buffer->size(), size_ * dTypeSize(dtype)));
  }
  calcStrides();
}

const Shape& Tensor::shape() const { return shape_; }

DataType Tensor::dtype() const { return dtype_; }

size_t Tensor::size() const { return size_; }

size_t Tensor::nbytes() const { return buffer_->size(); }

std::vector<ptrdiff_t> Tensor::strides() const { return strides_; }

bool Tensor::isContiguous() const {
  ptrdiff_t expected_stride = 1;
  for (int i = shape_.ndim() - 1; i >= 0; --i) {
    if (strides_[i] != expected_stride) {
      return false;
    }
    expected_stride *= shape_[i];
  }
  return true;
}

Result<TensorRef, std::string> Tensor::toDeviceDense(memory::DeviceAllocator* allocator) {
  auto dtype_size = dTypeSize(dtype_);
  DECLARE_OR_RETURN(new_buffer, memory::Buffer::create(size_ * dtype_size, allocator));
  new_buffer->copyFrom(buffer_, 0, offset_ * dtype_size, size_ * dtype_size);
  auto new_tensor = std::shared_ptr<Tensor>(new Tensor(dtype_, shape_, new_buffer));
  return Ok(new_tensor);
}

Result<TensorRef, std::string> Tensor::toDevicePreserveLayout(memory::DeviceAllocator* allocator) {
  if (buffer_->devType() == allocator->devType()) {
    return Ok(shared_from_this());
  }
  auto dtype_size = dTypeSize(dtype_);
  DECLARE_OR_RETURN(new_buffer, memory::Buffer::create(buffer_->size(), allocator));
  new_buffer->copyFrom(buffer_);
  auto new_tensor = std::shared_ptr<Tensor>(new Tensor(dtype_, shape_, new_buffer));
  new_tensor->offset_ = offset_;
  new_tensor->strides_ = strides_;
  new_tensor->size_ = size_;

  return Ok(new_tensor);
}

Result<TensorRef, std::string> Tensor::toDevice(memory::DeviceAllocator* allocator,
                                                bool preserveLayout) {
  CHECK_NE(buffer_, nullptr);
  CHECK_NE(allocator, nullptr);
  CHECK_NE(buffer_->devType(), DeviceType::kDeviceUnknown);
  // only continuous or non-slice tensor can be copied for now
  CHECK(isContiguous() || buffer_->size() == size_ * dTypeSize(dtype_))
      << "Only contiguous tensors are supported for now.";

  if (preserveLayout) {
    return toDevicePreserveLayout(allocator);
  } else {
    return toDeviceDense(allocator);
  }
}

Result<TensorRef, std::string> Tensor::toDevice(DeviceType dev_type, bool preserveLayout) {
  return toDevice(memory::getDefaultDeviceAllocator(dev_type), preserveLayout);
}

void Tensor::copyFrom(const TensorRef& src) {
  CHECK(shape_ == src->shape()) << "Copy failed: tensor shapes do not match.";
  CHECK(dtype_ == src->dtype()) << "Copy failed: tensor data types do not match.";
  CHECK(strides_ == src->strides()) << "Copy failed: tensor strides do not match.";
  CHECK(src->isContiguous() && isContiguous())
      << "Copy failed: src or dst tensor is not contiguous.";
  auto dtype_size = dTypeSize(dtype_);
  buffer_->copyFrom(src->buffer_, src->offset_ * dtype_size, offset_ * dtype_size,
                    size_ * dtype_size);
}

TensorRef Tensor::slice(int dim, int64_t start, int64_t end) const {
  CHECK(dim >= 0 && dim < static_cast<int>(shape_.ndim()))
      << "Slice failed: dimension out of range.";
  CHECK(start >= 0 && end <= shape_[dim] && start < end)
      << "Slice failed: invalid start or end indices.";

  Shape new_shape = shape_;
  new_shape[dim] = end - start;
  // TODO more tidy way to implement this
  auto new_tensor = std::shared_ptr<Tensor>(new Tensor(dtype_, new_shape, buffer_));
  int64_t new_offset = start * strides_[dim] + offset_;
  new_tensor->offset_ = new_offset;
  new_tensor->strides_ = strides_;
  new_tensor->size_ = new_shape.numel();
  return new_tensor;
}

TensorRef Tensor::reshape(const Shape& new_shape) const {
  CHECK_EQ(shape_.numel(), new_shape.numel())
      << "Reshape failed: number of elements does not match.";
  auto new_tensor = std::shared_ptr<Tensor>(new Tensor(dtype_, new_shape, buffer_));
  new_tensor->offset_ = offset_;
  return new_tensor;
}

TensorRef Tensor::permute(const std::vector<size_t>& new_order) const {
  CHECK_EQ(new_order.size(), shape_.ndim())
      << "Permute failed: new order size does not match tensor dimensions.";

  Shape new_shape(shape_.ndim());
  std::vector<ptrdiff_t> new_strides(shape_.ndim());
  for (size_t i = 0; i < new_order.size(); ++i) {
    CHECK_LT(new_order[i], shape_.ndim()) << "Permute failed: index out of range.";
    new_shape[i] = shape_[new_order[i]];
    new_strides[i] = strides_[new_order[i]];
  }

  auto new_tensor = std::shared_ptr<Tensor>(new Tensor(dtype_, new_shape, buffer_));
  new_tensor->offset_ = offset_;
  new_tensor->strides_ = new_strides;
  return new_tensor;
}

void Tensor::calcStrides() {
  size_t ndim = shape_.ndim();
  strides_.resize(ndim);
  strides_[ndim - 1] = 1;
  std::partial_sum(shape_.rbegin(), shape_.rend() - 1, strides_.rbegin() + 1,
                   std::multiplies<ptrdiff_t>());
}

}  // namespace ginfer::core::tensor