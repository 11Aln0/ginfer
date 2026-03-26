#include <glog/logging.h>
#include <sys/sysinfo.h>
#include <cstring>
#include "allocator.h"
#include "ginfer/common/errors.h"

namespace ginfer::core::memory {

CPUDeviceAllocator::CPUDeviceAllocator() : DeviceAllocator(DeviceType::kDeviceCPU) {}

Result<void*, std::string> CPUDeviceAllocator::doAlloc(size_t size) {
  if (size == 0) {
    LOG(WARNING) << "Try to allocate 0 bytes.";
    return Ok((void*)nullptr);
  }
  void* ptr = std::malloc(size);
  RETURN_ERR_ON(ptr == nullptr, "cpu malloc failed for size {}", size);
  return Ok(ptr);
}

void CPUDeviceAllocator::doFree(void* ptr, size_t size) {
  (void)size;
  std::free(ptr);
}

void CPUDeviceAllocator::memcpy(
    const void* src, void* dst, size_t size, MemcpyKind kind, void* stream, bool sync) const {
  (void)kind;
  (void)stream;
  (void)sync;
  std::memcpy(dst, src, size);
}

DeviceMemInfo CPUDeviceAllocator::getMemInfo() const {
  struct sysinfo info;
  if (sysinfo(&info) == 0) {
    return {info.totalram * info.mem_unit, info.freeram * info.mem_unit};
  }
  return {0, 0};
}

}  // namespace ginfer::core::memory