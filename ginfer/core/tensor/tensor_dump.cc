#include "ginfer/core/tensor/tensor_dump.h"

#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <ostream>
#include <sstream>
#include <vector>

#include "ginfer/common/device.h"
#include "ginfer/core/memory/allocator.h"

namespace ginfer::core::tensor {
namespace {

std::string formatShape(const Shape& shape) {
  std::ostringstream os;
  os << "[";
  for (size_t i = 0; i < shape.ndim(); ++i) {
    if (i != 0) {
      os << ", ";
    }
    os << shape[i];
  }
  os << "]";
  return os.str();
}

std::string formatStrides(const std::vector<ptrdiff_t>& strides) {
  std::ostringstream os;
  os << "[";
  for (size_t i = 0; i < strides.size(); ++i) {
    if (i != 0) {
      os << ", ";
    }
    os << strides[i];
  }
  os << "]";
  return os.str();
}

const char* dataTypeName(DataType dtype) {
  switch (dtype) {
    case DataType::kDataTypeFloat32:
      return "Float32";
    case DataType::kDataTypeFloat16:
      return "Float16";
    case DataType::kDataTypeBFloat16:
      return "BFloat16";
    case DataType::kDataTypeInt64:
      return "Int64";
    case DataType::kDataTypeInt32:
      return "Int32";
    case DataType::kDataTypeInt8:
      return "Int8";
    case DataType::kDataTypeVoid:
    default:
      return "Void";
  }
}

template <typename T>
void appendValues(std::ostringstream& os, const TensorRef& tensor, size_t count) {
  const auto* data = tensor->data<T>();
  for (size_t i = 0; i < count; ++i) {
    if (i != 0) {
      os << ", ";
    }
    if constexpr (std::is_same_v<T, int8_t>) {
      os << static_cast<int>(data[i]);
    } else {
      os << data[i];
    }
  }
}

void appendRaw16Values(std::ostringstream& os, const TensorRef& tensor, size_t count) {
  const auto* data = tensor->data<uint16_t>();
  auto flags = os.flags();
  auto fill = os.fill();
  for (size_t i = 0; i < count; ++i) {
    if (i != 0) {
      os << ", ";
    }
    os << "0x" << std::hex << std::setw(4) << std::setfill('0') << data[i] << std::dec;
  }
  os.flags(flags);
  os.fill(fill);
}

void appendData(std::ostringstream& os, const TensorRef& tensor, size_t max_elements) {
  const size_t shown = std::min(max_elements, tensor->size());
  os << "  data: [";
  switch (tensor->dtype()) {
    case DataType::kDataTypeFloat32:
      appendValues<float>(os, tensor, shown);
      break;
    case DataType::kDataTypeFloat16:
    case DataType::kDataTypeBFloat16:
      appendRaw16Values(os, tensor, shown);
      break;
    case DataType::kDataTypeInt64:
      appendValues<int64_t>(os, tensor, shown);
      break;
    case DataType::kDataTypeInt32:
      appendValues<int32_t>(os, tensor, shown);
      break;
    case DataType::kDataTypeInt8:
      appendValues<int8_t>(os, tensor, shown);
      break;
    case DataType::kDataTypeVoid:
    default:
      os << "unsupported dtype for dump";
      break;
  }
  os << "] " << shown << "/" << tensor->size() << " elements shown\n";
}

std::string formatDeviceType(common::DeviceType dev_type) {
  std::ostringstream os;
  os << dev_type;
  return os.str();
}

}  // namespace

std::string dumpTensor(const TensorRef& tensor, const TensorDumpOptions& options) {
  std::ostringstream os;
  if (tensor == nullptr) {
    os << "Tensor { null }\n";
    return os.str();
  }

  os << "Tensor {\n";
  os << "  shape: " << formatShape(tensor->shape()) << "\n";
  os << "  dtype: " << dataTypeName(tensor->dtype()) << "\n";
  os << "  device: " << formatDeviceType(tensor->devType()) << "\n";
  os << "  size: " << tensor->size() << "\n";
  os << "  nbytes: " << tensor->nbytes() << "\n";
  os << "  strides: " << formatStrides(tensor->strides()) << "\n";
  os << "  contiguous: " << (tensor->isContiguous() ? "true" : "false") << "\n";

  if (!options.include_data) {
    os << "  data: skipped\n";
    os << "}\n";
    return os.str();
  }

  TensorRef readable = tensor;
  if (tensor->devType() != common::DeviceType::kDeviceCPU || !tensor->isContiguous()) {
    if (!options.copy_to_cpu) {
      os << "  data: skipped because tensor is not a contiguous CPU tensor\n";
      os << "}\n";
      return os.str();
    }

    auto cpu_res = tensor->toDevice(common::DeviceType::kDeviceCPU, memory::kDefault, false);
    if (!cpu_res.ok()) {
      os << "  data: failed to copy tensor to contiguous CPU buffer: " << cpu_res.err() << "\n";
      os << "}\n";
      return os.str();
    }
    readable = std::move(cpu_res).value();
    os << "  data_source: copied to contiguous CPU buffer\n";
  }

  appendData(os, readable, options.max_elements);
  os << "}\n";
  return os.str();
}

void printTensor(const TensorRef& tensor, const TensorDumpOptions& options, std::ostream& os) {
  os << dumpTensor(tensor, options);
}

}  // namespace ginfer::core::tensor
