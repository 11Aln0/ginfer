#pragma once

#include <iostream>

namespace ginfer::common {

enum class DeviceType { kDeviceUnknown = 0, kDeviceCPU = 1, kDeviceCUDA = 2, kDeviceROCM = 3 };

static const char* device_type_names[4] = {"DeviceUnknown", "DeviceCPU", "DeviceCUDA",
                                           "DeviceROCM"};

inline std::ostream& operator<<(std::ostream& os, DeviceType dev_type) {
  os << device_type_names[static_cast<int>(dev_type)];
  return os;
}

bool isHostDevice(DeviceType dev_type);

}  // namespace ginfer::common