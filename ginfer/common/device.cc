#include "device.h"

namespace ginfer::common {

bool isHostDevice(DeviceType dev_type) { return dev_type == DeviceType::kDeviceCPU; }

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

}  // namespace ginfer::common