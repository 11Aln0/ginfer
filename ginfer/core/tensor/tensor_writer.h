#pragma once

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <type_traits>

#include <glog/logging.h>

#include "ginfer/core/tensor/dtype.h"
#include "ginfer/core/tensor/tensor.h"

namespace ginfer::core::tensor {

template <typename T>
class TensorWriter {
 public:
  explicit TensorWriter(const TensorRef& tensor) : tensor_(tensor) {
    validateTensor();
    ptr_ = tensor_->data<T>();
    capacity_ = static_cast<size_t>(tensor_->shape()[0]);
  }

  template <typename InputIt>
  void extend(InputIt first, InputIt last) {
    const auto n = static_cast<size_t>(std::distance(first, last));
    ensureCapacity(n);
    for (auto it = first; it != last; ++it) {
      ptr_[size_++] = static_cast<T>(*it);
    }
  }

  void append(const T& value) {
    ensureCapacity(1);
    ptr_[size_++] = value;
  }

  T& back() const {
    CHECK(size_ > 0) << "TensorWriter back() requires non-empty writer";
    return ptr_[size_ - 1];
  }

  size_t size() const { return size_; }
  size_t capacity() const { return capacity_; }

 private:
  void validateTensor() const {
    CHECK(tensor_ != nullptr) << "TensorWriter requires a non-null tensor";
    CHECK(tensor_->devType() == DeviceType::kDeviceCPU) << "TensorWriter requires a CPU tensor";
    CHECK(tensor_->shape().ndim() == 1) << "TensorWriter requires a 1D tensor";
    CHECK(tensor_->isContiguous()) << "TensorWriter requires a contiguous tensor";
    CHECK(tensor_->dtype() ==
          (DataTypeOf<typename type::TypeOf<DeviceType::kDeviceCPU, T>::type>::dtype))
        << "TensorWriter tensor dtype mismatch";
  }

  void ensureCapacity(size_t n) const {
    CHECK(size_ + n <= capacity_) << "TensorWriter capacity exceeded: size " << size_ << " + n "
                                  << n << " > capacity " << capacity_;
  }

 private:
  TensorRef tensor_;
  T* ptr_ = nullptr;
  size_t capacity_ = 0;
  size_t size_ = 0;
};

template <typename T>
TensorWriter<T> bindTensor(const TensorRef& tensor) {
  return TensorWriter<T>(tensor);
}

template <typename T>
class TensorWriter2D {
 public:
  explicit TensorWriter2D(const TensorRef& tensor) : tensor_(tensor) {
    validateTensor();
    ptr_ = tensor_->data<T>();
    rows_ = static_cast<size_t>(tensor_->shape()[0]);
    cols_ = static_cast<size_t>(tensor_->shape()[1]);
  }

  template <typename InputIt>
  void appendRow(InputIt first, InputIt last, size_t padToLen, T padVal = static_cast<T>(-1)) {
    CHECK(rowCount_ < rows_) << "TensorWriter2D row capacity exceeded";
    CHECK(padToLen <= cols_) << "TensorWriter2D padToLen exceeds tensor width";

    const auto len = static_cast<size_t>(std::distance(first, last));
    CHECK(len <= padToLen) << "TensorWriter2D row length exceeds padToLen";

    T* rowPtr = ptr_ + rowCount_ * cols_;
    size_t i = 0;
    for (auto it = first; it != last; ++it, ++i) {
      rowPtr[i] = static_cast<T>(*it);
    }
    std::fill(rowPtr + i, rowPtr + padToLen, padVal);
    ++rowCount_;
  }

  size_t size() const { return rowCount_; }
  size_t rows() const { return rows_; }
  size_t cols() const { return cols_; }

 private:
  void validateTensor() const {
    CHECK(tensor_ != nullptr) << "TensorWriter2D requires a non-null tensor";
    CHECK(tensor_->devType() == DeviceType::kDeviceCPU) << "TensorWriter requires a CPU tensor";
    CHECK(tensor_->shape().ndim() == 2) << "TensorWriter2D requires a 2D tensor";
    CHECK(tensor_->isContiguous()) << "TensorWriter2D requires a contiguous tensor";
    CHECK(tensor_->dtype() ==
          (DataTypeOf<typename type::TypeOf<DeviceType::kDeviceCPU, T>::type>::dtype))
        << "TensorWriter2D tensor dtype mismatch";
  }

 private:
  TensorRef tensor_;
  T* ptr_ = nullptr;
  size_t rows_ = 0;
  size_t cols_ = 0;
  size_t rowCount_ = 0;
};

template <typename T>
TensorWriter2D<T> bindTensor2D(const TensorRef& tensor) {
  return TensorWriter2D<T>(tensor);
}

}  // namespace ginfer::core::tensor
