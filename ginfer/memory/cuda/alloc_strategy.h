#pragma once

#include <concurrentqueue.h>
#include <cstddef>
#include <unordered_map>
#include <vector>

namespace ginfer::memory::cuda {

// struct AllocStrategy {
//   virtual ~AllocStrategy() = default;

//   virtual void* alloc(size_t size) const = 0;

//   virtual void free(void* ptr) const = 0;
// };

struct DefaultAllocStrategy {
  void* alloc(size_t size);

  void free(void* ptr, size_t size);
};

struct PooledAllocStrategy {
 private:
  static constexpr size_t MIN_BLOCK_SIZE_SHIFT = 12;                      // 2^12B = 4KB
  static constexpr size_t MIN_BLOCK_SIZE = 1ULL << MIN_BLOCK_SIZE_SHIFT;  // 4KB
  static constexpr size_t MAX_BLOCK_SIZE = 1ULL << 21;                    // 2MB
  static constexpr size_t NUM_BUCKETS = 10;                               // From 4KB to 2MB

  // size_t limited_pool_size_;  // byte
  // size_t used_;               // byte

  std::vector<moodycamel::ConcurrentQueue<void*>> free_blocks_;

 public:
  void* alloc(size_t size);

  void free(void* ptr, size_t size);

  ~PooledAllocStrategy();

 private:
  size_t roundUpToBlockSize(size_t size) const;
  int getBucketIndex(size_t aligned_size) const;
  void* getOrAllocBlock(size_t size);

 public:
  PooledAllocStrategy();
};

}  // namespace ginfer::memory::cuda