#pragma once

#include <cstddef>
#include <cstdint>
#include <numeric>
#include <vector>

namespace ginfer::core::tensor {

class Shape {
 public:
  explicit Shape() = delete;

  explicit Shape(int ndim) : dims_(ndim, 0) {}

  explicit Shape(const std::initializer_list<int64_t> dims) : dims_(dims) {}

  explicit Shape(const std::vector<int64_t>& dims) : dims_(dims) {}

  explicit Shape(std::vector<int64_t>&& dims) : dims_(std::move(dims)) {}

  size_t ndim() const { return dims_.size(); }

  int64_t numel() const {
    if (dims_.empty()) return 0;
    return std::accumulate(dims_.begin(), dims_.end(), 1, std::multiplies<int64_t>());
  }

  int64_t operator[](int64_t idx) const { return dims_.at(idx); }
  int64_t& operator[](int64_t idx) { return dims_.at(idx); }

  bool operator==(const Shape& other) const { return this->dims_ == other.dims_; }

  auto begin() const { return dims_.begin(); }
  auto end() const { return dims_.end(); }
  auto rbegin() const { return dims_.rbegin(); }
  auto rend() const { return dims_.rend(); }

 private:
  std::vector<int64_t> dims_;
};

}  // namespace ginfer::core::tensor