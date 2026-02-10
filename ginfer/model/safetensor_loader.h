#pragma once
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include "ginfer/tensor/tensor.h"

namespace ginfer::model {

struct SafeTensorMeta {
  std::string dtype;
  std::vector<int64_t> shape;
  size_t data_offset_begin;
  size_t data_offset_end;
};

class SafeTensorLoader {
 public:
  SafeTensorLoader() = default;

  ~SafeTensorLoader();

  void load(const std::string& filepath);

  std::shared_ptr<tensor::Tensor> getTensor(const std::string& name) const;

 private:
  void parseHeader(const std::string& json_str);

  tensor::DataType parseDataType(const std::string& dtype_str) const;

 private:
  char* mapped_data_ = nullptr;
  const char* data_base_ = nullptr;
  size_t file_size_ = 0;
  std::unordered_map<std::string, SafeTensorMeta> metadata_;
};

}  // namespace ginfer::model
