#include "device.h"

namespace ginfer::common {

bool isHostDevice(DeviceType dev_type) { return dev_type == DeviceType::kDeviceCPU; }

};  // namespace ginfer::common