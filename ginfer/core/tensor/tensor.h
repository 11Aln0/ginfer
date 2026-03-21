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

class Tensor : public std::enable_shared_from_this<Tensor> {
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

  bool isContiguous() const;

  Result<TensorRef, std::string> toDevice(DeviceType dev_type, bool preserveLayout = true);

  Result<TensorRef, std::string> toDevice(memory::DeviceAllocator* allocator,
                                          bool preserveLayout = true);

  void copyFrom(const TensorRef& src);

  template <typename T>
  T* data() {
    return reinterpret_cast<T*>(buffer_->ptr() + offset_ * dTypeSize(dtype_));
  }

  template <typename T>
  const T* data() const {
    return reinterpret_cast<const T*>(buffer_->ptr() + offset_ * dTypeSize(dtype_));
  }

  TensorRef slice(int dim, int64_t start, int64_t end) const;
  TensorRef reshape(const Shape& new_shape) const;
  TensorRef permute(const std::vector<size_t>& new_order) const;

 private:
  explicit Tensor(DataType dtype, Shape shape, std::shared_ptr<memory::Buffer> buffer);

  // use default allocator to create buffer
  explicit Tensor(DataType dtype, Shape shape, DeviceType dev_type);

  explicit Tensor(DataType dtype, Shape shape, memory::DeviceAllocator* allocator);

  void calcStrides();

  Result<TensorRef, std::string> toDeviceDense(memory::DeviceAllocator* allocator);
  Result<TensorRef, std::string> toDevicePreserveLayout(memory::DeviceAllocator* allocator);

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