#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include "ginfer/common/errors.h"
#include "ginfer/core/memory/allocator.h"
#include "ginfer/core/memory/allocator_factory.h"
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

  static Result<TensorRef, std::string> create(DataType dtype,
                                               Shape shape,
                                               DeviceType dev_type,
                                               uint8_t alloc_flags = memory::kDefault);

  static Result<TensorRef, std::string> create(DataType dtype,
                                               Shape shape,
                                               DeviceAllocator* allocator);

  explicit Tensor() = delete;

  const Shape& shape() const;

  DataType dtype() const;

  DeviceType devType() const;

  size_t size() const;

  size_t nbytes() const;

  std::vector<ptrdiff_t> strides() const;

  bool isContiguous() const;

  Result<TensorRef, std::string> toDevice(DeviceType dev_type,
                                          uint8_t alloc_flags = memory::kDefault,
                                          bool preserveLayout = true,
                                          bool async = false);

  Result<TensorRef, std::string> toDevice(memory::DeviceAllocator* allocator,
                                          bool preserveLayout = true,
                                          bool async = false);

  void copyFrom(const TensorRef& src, bool async = false);

  template <typename T = void>
  T* data() {
    return reinterpret_cast<T*>(buffer_->ptr() + offset_ * dTypeSize(dtype_));
  }

  template <typename T = void>
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

  Result<TensorRef, std::string> toDeviceDense(memory::DeviceAllocator* allocator,
                                               bool async = false);
  Result<TensorRef, std::string> toDevicePreserveLayout(memory::DeviceAllocator* allocator,
                                                        bool async = false);

 private:
  DataType dtype_ = DataType::kDataTypeVoid;
  std::shared_ptr<memory::Buffer> buffer_ = nullptr;
  Shape shape_;
  std::vector<ptrdiff_t> strides_;
  int64_t offset_ = 0;
  size_t size_ = 0;
  // Layout layout_ = Layout::kLayoutRowMajor;
};

}  // namespace ginfer::core::tensor