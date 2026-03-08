#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include "ginfer/common/errors.h"
#include "ginfer/core/memory/allocator.h"
#include "ginfer/core/memory/buffer.h"
#include "ginfer/core/tensor/dtype.h"
#include "ginfer/core/tensor/layout.h"
#include "ginfer/core/tensor/shape.h"

namespace ginfer::core::tensor {

class Tensor;

using ginfer::core::memory::DeviceAllocator;
using ginfer::core::memory::DeviceType;
using TensorRef = std::shared_ptr<tensor::Tensor>;

class Tensor {
 public:
  static Result<TensorRef, std::string> create(DataType dtype,
                                               Shape shape,
                                               std::shared_ptr<memory::Buffer> buffer);

  static Result<TensorRef, std::string> create(DataType dtype, Shape shape, DeviceType dev_type);

  static Result<TensorRef, std::string> create(DataType dtype,
                                               Shape shape,
                                               DeviceAllocator* allocator);

  explicit Tensor() = delete;

  const Shape& shape() const;

  DataType dtype() const;

  size_t size() const;

  size_t nbytes() const;

  std::vector<ptrdiff_t> strides() const;

  Result<void, std::string> toDevice(DeviceType dev_type);

  Result<void, std::string> toDevice(memory::DeviceAllocator* allocator);

  void copyFrom(const Tensor& src);

  template <typename T>
  T* data() {
    return reinterpret_cast<T*>(buffer_->ptr() + offset_ * dTypeSize(dtype_));
  }

  template <typename T>
  const T* data() const {
    return reinterpret_cast<const T*>(buffer_->ptr() + offset_ * dTypeSize(dtype_));
  }

  std::shared_ptr<Tensor> slice(int dim, int64_t start, int64_t end) const;
  std::shared_ptr<Tensor> reshape(const Shape& new_shape) const;
  std::shared_ptr<Tensor> permute(const std::vector<size_t>& new_order) const;

 private:
  explicit Tensor(DataType dtype, Shape shape, std::shared_ptr<memory::Buffer> buffer);

  // use default allocator to create buffer
  explicit Tensor(DataType dtype, Shape shape, DeviceType dev_type);

  explicit Tensor(DataType dtype, Shape shape, memory::DeviceAllocator* allocator);

  void calcStrides();

 private:
  DataType dtype_ = DataType::kDataTypeUnknown;
  std::shared_ptr<memory::Buffer> buffer_ = nullptr;
  Shape shape_;
  std::vector<ptrdiff_t> strides_;
  int64_t offset_ = 0;
  size_t size_ = 0;
  // Layout layout_ = Layout::kLayoutRowMajor;
};

}  // namespace ginfer::core::tensor