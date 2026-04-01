#include "ginfer/model/loader/safetensor_loader.h"
#include <nlohmann/json.hpp>
#include "ginfer/common/errors.h"

namespace ginfer::model {

SafeTensorLoader::~SafeTensorLoader() {
  if (mapped_data_ && mapped_data_ != MAP_FAILED) {
    munmap(mapped_data_, file_size_);
  }
}

void SafeTensorLoader::load(const std::string& filepath) {
  int fd = open(filepath.c_str(), O_RDONLY);
  CHECK_THROW(fd != -1, "Failed to open file: {}", filepath);

  // mmap file
  file_size_ = lseek(fd, 0, SEEK_END);
  mapped_data_ = static_cast<char*>(mmap(nullptr, file_size_, PROT_READ, MAP_PRIVATE, fd, 0));
  close(fd);
  CHECK_THROW(mapped_data_ != MAP_FAILED, "Failed to map file: {}", filepath);

  // get header size
  uint64_t header_size = *reinterpret_cast<uint64_t*>(mapped_data_);
  data_base_ = mapped_data_ + sizeof(uint64_t) + header_size;

  std::string json_str(mapped_data_ + sizeof(uint64_t), header_size);
  parseHeader(json_str);
}

void SafeTensorLoader::parseHeader(const std::string& json_str) {
  auto json = nlohmann::json::parse(json_str);
  for (auto& [key, meta] : json.items()) {
    if (key == "__metadata__") continue;
    SafeTensorMeta tensor_meta;
    tensor_meta.dtype = meta["dtype"].get<std::string>();
    tensor_meta.shape = meta["shape"].get<std::vector<int64_t>>();
    tensor_meta.data_offset_begin = meta["data_offsets"][0].get<size_t>();
    tensor_meta.data_offset_end = meta["data_offsets"][1].get<size_t>();
    metadata_[key] = std::move(tensor_meta);
  }
}

core::tensor::DataType SafeTensorLoader::parseDataType(const std::string& dtype_str) const {
  using core::tensor::DataType;
  if (dtype_str == "I8") {
    return DataType::kDataTypeInt8;
  } else if (dtype_str == "BF16") {
    return DataType::kDataTypeBFloat16;
  } else if (dtype_str == "F16") {
    return DataType::kDataTypeFloat16;
  } else if (dtype_str == "F32") {
    return DataType::kDataTypeFloat32;
  } else {
    CHECK_THROW(false, "Unsupported data type: {}", dtype_str);
  }
}

core::tensor::TensorRef SafeTensorLoader::getTensor(const std::string& name) const {
  using namespace core;
  auto it = metadata_.find(name);
  CHECK_THROW(it != metadata_.end(), "Tensor not found: {}", name);
  const auto& meta = it->second;
  tensor::DataType dtype = parseDataType(meta.dtype);
  const char* data_ptr = data_base_ + meta.data_offset_begin;
  size_t num_elements = 1;
  for (size_t dim : meta.shape) {
    num_elements *= dim;
  }
  size_t element_size = tensor::dTypeSize(dtype);
  size_t expected_size = num_elements * element_size;
  CHECK_THROW(expected_size == meta.data_offset_end - meta.data_offset_begin,
              "Data size mismatch for tensor {}: expected {}, actual {}", name, expected_size,
              meta.data_offset_end - meta.data_offset_begin);
  auto buf_res =
      memory::Buffer::create(expected_size, (std::byte*)data_ptr, memory::DeviceType::kDeviceCPU);
  CHECK_THROW(buf_res.ok(), "Failed to create buffer for tensor {}: {}", name, buf_res.err());
  auto res = tensor::Tensor::create(dtype, tensor::Shape(meta.shape), buf_res.value());
  CHECK_THROW(res.ok(), "Failed to create tensor {}: {}", name, res.err());
  return res.value();
}

}  // namespace ginfer::model