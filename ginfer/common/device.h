#pragma once

#include <cuda_runtime.h>
#include <iostream>
#include <memory>

namespace ginfer::common {

enum class DeviceType { kDeviceUnknown = 0, kDeviceCPU = 1, kDeviceCUDA = 2, kDeviceROCM = 3 };

static const char* device_type_names[4] = {"DeviceUnknown", "DeviceCPU", "DeviceCUDA",
                                           "DeviceROCM"};

inline std::ostream& operator<<(std::ostream& os, DeviceType dev_type) {
  os << device_type_names[static_cast<int>(dev_type)];
  return os;
}

bool isHostDevice(DeviceType dev_type);

class DeviceContext {
 public:
  static std::unique_ptr<DeviceContext> create(DeviceType dev_type);

  explicit DeviceContext(DeviceType dev_type);

  virtual ~DeviceContext() = default;

  DeviceType getDeviceType() const;

 private:
  DeviceType dev_type_;
};

class CPUDeviceContext : public DeviceContext {
 public:
  CPUDeviceContext();
};

class CUDADeviceContext : public DeviceContext {
 public:
  CUDADeviceContext(cudaStream_t stream = nullptr);

  ~CUDADeviceContext() override;

  cudaStream_t getStream() const;

 private:
  cudaStream_t stream_ = nullptr;
};

template <DeviceType dev_type>
struct DeviceContextType {
  using type = DeviceContext;
};

template <>
struct DeviceContextType<DeviceType::kDeviceCPU> {
  using type = CPUDeviceContext;
};

template <>
struct DeviceContextType<DeviceType::kDeviceCUDA> {
  using type = CUDADeviceContext;
};

}  // namespace ginfer::common