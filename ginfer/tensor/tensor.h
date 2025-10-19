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

  explicit Tensor(DType dtype, Shape shape, std::shared_ptr<memory::Buffer> buffer);

  explicit Tensor(DType dtype, Shape shape, DeviceType dev_type);

  const Shape& shape() const;

  DType dtype() const;

  size_t size() const;

  size_t nbytes() const;

  std::vector<size_t> strides() const;

  void toDev(DeviceType dev_type);

 private:
  DType dtype_ = DType::kDTypeUnknown;
  std::shared_ptr<memory::Buffer> buffer_ = nullptr;
  Shape shape_;
  size_t size_ = 0;
};

}  // namespace ginfer::tensor