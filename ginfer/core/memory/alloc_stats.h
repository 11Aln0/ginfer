#pragma once
#include <cstddef>

namespace ginfer::core::memory {

struct AllocatorStats {
  size_t live_bytes = 0;
  size_t peak_live_bytes = 0;
  size_t reserved_bytes = 0;
  size_t peak_reserved_bytes = 0;
};

class AllocatorStatsTracker {
 public:
  void onAlloc(size_t size);

  void onReserve(size_t size);

  void onRelease(size_t size);

  void onFree(size_t size);

  void reset();

  const AllocatorStats& getStats();

 private:
  AllocatorStats stats_;
};

}  // namespace ginfer::core::memory