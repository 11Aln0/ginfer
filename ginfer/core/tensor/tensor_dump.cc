#include "ginfer/core/tensor/tensor_dump.h"

#include <algorithm>
#include <cstdint>
#include <ostream>
#include <sstream>
#include <vector>

#include "ginfer/common/device.h"
#include "ginfer/common/type.h"
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

template <DataType dtype>
void appendValues(std::ostringstream& os, const TensorRef& tensor, size_t count) {
  using SrcType = typename TypeOf<dtype>::type;
  using DisplayType = typename DisplayTypeOf<SrcType>::type;

  const auto* data = tensor->data<SrcType>();
  for (size_t i = 0; i < count; ++i) {
    if (i != 0) {
      os << ", ";
    }
    os << type::TypeConverter<SrcType, DisplayType>::convert(data[i]);
  }
}

void appendData(std::ostringstream& os, const TensorRef& tensor, size_t max_elements) {
  const size_t shown = std::min(max_elements, tensor->size());
  os << "  data: [";
  switch (tensor->dtype()) {
    case DataType::kDataTypeFloat32:
      appendValues<DataType::kDataTypeFloat32>(os, tensor, shown);
      break;
    case DataType::kDataTypeFloat16:
      appendValues<DataType::kDataTypeFloat16>(os, tensor, shown);
      break;
    case DataType::kDataTypeBFloat16:
      appendValues<DataType::kDataTypeBFloat16>(os, tensor, shown);
      break;
    case DataType::kDataTypeInt64:
      appendValues<DataType::kDataTypeInt64>(os, tensor, shown);
      break;
    case DataType::kDataTypeInt32:
      appendValues<DataType::kDataTypeInt32>(os, tensor, shown);
      break;
    case DataType::kDataTypeInt8:
      appendValues<DataType::kDataTypeInt8>(os, tensor, shown);
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

    auto cpu_res = tensor->toDevice(common::DeviceType::kDeviceCPU, memory::kDefault);
    if (!cpu_res.ok()) {
      os << "  data: failed to copy tensor to CPU buffer: " << cpu_res.err() << "\n";
      os << "}\n";
      return os.str();
    }
    readable = std::move(cpu_res).value();
    os << "  data_source: copied to CPU buffer\n";
  }

  appendData(os, readable, options.max_elements);
  os << "}\n";
  return os.str();
}

void printTensor(const TensorRef& tensor, const TensorDumpOptions& options, std::ostream& os) {
  os << dumpTensor(tensor, options);
}

}  // namespace ginfer::core::tensor
