#pragma once

#include <cuda_runtime.h>
#include <iostream>
#include <memory>
#include <cstddef>
#include <cstdint>

namespace ginfer::common {

enum class DeviceType {
  kDeviceUnknown = 0,
  kDeviceCPU = 1,
  kDeviceCUDA = 2,
  kDeviceROCM = 3,
  kDeviceMax = 4
};

static const char* device_type_names[4] = {"DeviceUnknown", "DeviceCPU", "DeviceCUDA",
                                           "DeviceROCM"};

inline std::ostream& operator<<(std::ostream& os, DeviceType dev_type) {
  os << device_type_names[static_cast<int>(dev_type)];
  return os;
}

bool isHostDevice(DeviceType dev_type);

class DeviceContext {
 public:
  static std::shared_ptr<DeviceContext> create(DeviceType dev_type);

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
  static constexpr size_t kDefaultWorkspaceSize = 1 << 22;

  CUDADeviceContext(cudaStream_t stream = nullptr,
                    size_t workspace_size = kDefaultWorkspaceSize);

  ~CUDADeviceContext() override;

  cudaStream_t getStream() const;
  void* getWorkspace() const;
  size_t getWorkspaceSize() const;

 private:
  cudaStream_t stream_ = nullptr;
  void* workspace_ = nullptr;
  size_t workspace_size_ = 0;
};

template <DeviceType Device>
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