#pragma once

#include <concurrentqueue.h>
#include <glog/logging.h>
#include <cstddef>
#include <unordered_map>
#include <vector>
#include "ginfer/common/errors.h"

namespace ginfer::core::memory {

template <typename BaseAllocator>
using DefaultAlllocStrategy = BaseAllocator;

template <typename BaseAllocator>
class PooledAllocStrategy : public BaseAllocator {
 public:
  PooledAllocStrategy() : free_blocks_(NUM_BUCKETS) {}

  Result<void*, std::string> doAlloc(size_t size) override { return getOrAllocBlock(size); }

  void doFree(void* ptr, size_t size) override {
    int bucket_index = getBucketIndex(roundUpToBlockSize(size));
    free_blocks_[bucket_index].enqueue(ptr);
  }

  virtual ~PooledAllocStrategy() {
    for (size_t i = 0; i < NUM_BUCKETS; ++i) {
      void* ptr;
      size_t block_size = 1ULL << (i + MIN_BLOCK_SIZE_SHIFT);
      while (free_blocks_[i].try_dequeue(ptr)) {
        BaseAllocator::doFree(ptr, block_size);
        this->onRelease(block_size);
      }
    }
  }

 private:
  static constexpr size_t MIN_BLOCK_SIZE_SHIFT = 12;                      // 2^12B = 4KB
  static constexpr size_t MIN_BLOCK_SIZE = 1ULL << MIN_BLOCK_SIZE_SHIFT;  // 4KB
  static constexpr size_t MAX_BLOCK_SIZE = 1ULL << 21;                    // 2MB
  static constexpr size_t NUM_BUCKETS = 10;                               // From 4KB to 2MB

  std::vector<moodycamel::ConcurrentQueue<void*>> free_blocks_;

 private:
  size_t roundUpToBlockSize(size_t size) const {
    if (size < MIN_BLOCK_SIZE) {
      return MIN_BLOCK_SIZE;
    }
    return 1ULL << (64 - __builtin_clzll(size - 1));
  }

  int getBucketIndex(size_t aligned_size) const {
    CHECK(aligned_size <= MAX_BLOCK_SIZE && aligned_size > 0)
        << "Requested size " << aligned_size << " should be in (0, " << MAX_BLOCK_SIZE << "]";
    return static_cast<int>(63 - __builtin_clzll(aligned_size)) - MIN_BLOCK_SIZE_SHIFT;
  }

  Result<void*, std::string> getOrAllocBlock(size_t size) {
    size_t aligned_size = roundUpToBlockSize(size);
    int bucket_index = getBucketIndex(aligned_size);
    void* p = nullptr;
    if (free_blocks_[bucket_index].try_dequeue(p)) {
      DLOG(INFO) << "Reusing block of size " << aligned_size / 1024.0 << " KB.";
    } else {
      auto res = BaseAllocator::doAlloc(aligned_size);
      RETURN_ON_ERR(res);
      this->onReserve(aligned_size);
      DLOG(INFO) << "Allocated new block of size " << aligned_size / 1024.0 << " KB.";
      p = res.value();
    }
    return Ok(p);
  }
};

}  // namespace ginfer::core::memory