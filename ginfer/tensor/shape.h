#pragma once

#include <cstddef>
#include <cstdint>
#include <numeric>
#include <vector>

namespace ginfer::tensor {

class Shape {
 public:
  explicit Shape() = delete;

  explicit Shape(const std::initializer_list<int64_t> dims) : dims_(dims){};

  explicit Shape(const std::vector<int64_t>& dims) : dims_(dims){};

  explicit Shape(std::vector<int64_t>&& dims) : dims_(std::move(dims)){};

  size_t ndim() const { return dims_.size(); }

  int64_t numel() const {
    if (dims_.empty()) return 0;
    return std::accumulate(dims_.begin(), dims_.end(), 1, std::multiplies<int64_t>());
  }

  int64_t operator[](size_t idx) const { return dims_[idx]; }
  int64_t& operator[](size_t idx) { return dims_[idx]; }

  bool operator==(const Shape& other) const {
    if (ndim() != other.ndim()) return false;
    for (size_t i = 0; i < ndim(); ++i) {
      if (dims_[i] != other.dims_[i]) return false;
    }
    return true;
  }

  auto begin() const { return dims_.begin(); }
  auto end() const { return dims_.end(); }
  auto rbegin() const { return dims_.rbegin(); }
  auto rend() const { return dims_.rend(); }

 private:
  std::vector<int64_t> dims_;
};

}  // namespace ginfer::tensor