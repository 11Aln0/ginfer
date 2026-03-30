#include "ginfer/core/memory/allocator.h"
#include "ginfer/core/memory/alloc_stats.h"

#include <glog/logging.h>

namespace ginfer::core::memory {

Result<void*, std::string> DeviceAllocator::alloc(size_t size) {
  auto res = doAlloc(size);
  RETURN_ON_ERR(res);
  onAlloc(size);
  return res;
}

void DeviceAllocator::free(void* ptr, size_t size) {
  doFree(ptr, size);
  onFree(size);
}

void DeviceAllocator::setStream(void* stream) { stream_ = stream; }

void AllocatorStatsTracker::onAlloc(size_t size) {
  stats_.live_bytes += size;
  if (stats_.live_bytes > stats_.peak_live_bytes) {
    stats_.peak_live_bytes = stats_.live_bytes;
  }
}

void AllocatorStatsTracker::onReserve(size_t size) {
  stats_.reserved_bytes += size;
  if (stats_.reserved_bytes > stats_.peak_reserved_bytes) {
    stats_.peak_reserved_bytes = stats_.reserved_bytes;
  }
}

void AllocatorStatsTracker::onRelease(size_t size) {
  CHECK_LE(size, stats_.reserved_bytes) << "Released bytes exceed reserved bytes.";
  stats_.reserved_bytes -= size;
}

void AllocatorStatsTracker::onFree(size_t size) {
  CHECK_LE(size, stats_.live_bytes) << "Freed bytes exceed live bytes.";
  stats_.live_bytes -= size;
}

void AllocatorStatsTracker::reset() { stats_ = {}; }

const AllocatorStats& AllocatorStatsTracker::getStats() { return stats_; }

}  // namespace ginfer::core::memory