#include "device.h"

namespace ginfer::common {

bool isHostDevice(DeviceType dev_type) { return dev_type == DeviceType::kDeviceCPU; }

std::shared_ptr<DeviceContext> DeviceContext::create(DeviceType dev_type) {
  switch (dev_type) {
    case DeviceType::kDeviceCPU:
      return std::make_shared<CPUDeviceContext>();
    case DeviceType::kDeviceCUDA:
      return std::make_shared<CUDADeviceContext>();
    default:
      throw std::runtime_error("Unsupported device type for DeviceContext creation.");
  }
}

DeviceContext::DeviceContext(DeviceType dev_type) : dev_type_(dev_type) {}

DeviceType DeviceContext::getDeviceType() const { return dev_type_; }

CPUDeviceContext::CPUDeviceContext() : DeviceContext(DeviceType::kDeviceCPU) {}

CUDADeviceContext::CUDADeviceContext(cudaStream_t stream)
    : DeviceContext(DeviceType::kDeviceCUDA), stream_(stream) {}

CUDADeviceContext::~CUDADeviceContext() {
  if (stream_) {
    cudaStreamDestroy(stream_);
  }
}

cudaStream_t CUDADeviceContext::getStream() const { return stream_; }

}  // namespace ginfer::common