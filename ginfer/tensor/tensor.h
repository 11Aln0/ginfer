#pragma once

#include <memory>
#include <vector>

#include "ginfer/memory/allocator.h"
#include "ginfer/memory/buffer.h"
#include "ginfer/tensor/dtype.h"
#include "ginfer/tensor/shape.h"

namespace ginfer::tensor {

using ginfer::memory::DeviceType;

class Tensor {
 public:
  explicit Tensor() = delete;

  explicit Tensor(DataType dtype, Shape shape, std::shared_ptr<memory::Buffer> buffer);

  explicit Tensor(DataType dtype, Shape shape, DeviceType dev_type);

  const Shape& shape() const;

  DataType dtype() const;

  size_t size() const;

  size_t nbytes() const;

  std::vector<size_t> strides() const;

  void toDevice(DeviceType dev_type);

  template <typename T>
  T* data() {
    return static_cast<T*>(buffer_->ptr());
  }

  template <typename T>
  const T* data() const {
    return static_cast<const T*>(buffer_->ptr());
  }

 private:
  DataType dtype_ = DataType::kDataTypeUnknown;
  std::shared_ptr<memory::Buffer> buffer_ = nullptr;
  Shape shape_;
  size_t size_ = 0;
};

}  // namespace ginfer::tensor