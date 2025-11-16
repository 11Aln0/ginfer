#include <cuda_runtime.h>
#include <glog/logging.h>
#include "ginfer/memory/cuda/alloc_strategy.h"

namespace ginfer::memory::cuda {

PooledAllocStrategy::PooledAllocStrategy() : free_blocks_(NUM_BUCKETS) {}

PooledAllocStrategy::~PooledAllocStrategy() {
  for (size_t i = 0; i < NUM_BUCKETS; ++i) {
    void* ptr;
    while (free_blocks_[i].try_dequeue(ptr)) {
      cudaError_t err = cudaFree(ptr);
      CHECK(err == cudaSuccess) << "Failed to free CUDA memory block";
    }
  }
}

size_t PooledAllocStrategy::roundUpToBlockSize(size_t size) const {
  if (size < MIN_BLOCK_SIZE) {
    return MIN_BLOCK_SIZE;
  }
  return 1ULL << (64 - __builtin_clzll(size - 1));
}

int PooledAllocStrategy::getBucketIndex(size_t size) const {
  CHECK(size <= MAX_BLOCK_SIZE && size > 0) << "Requested size " << size << " should be in (0, " << MAX_BLOCK_SIZE << "]";
  return static_cast<int>(63 - __builtin_clzll(size)) - MIN_BLOCK_SIZE_SHIFT;
}

void* PooledAllocStrategy::getOrAllocBlock(size_t size) {
  size_t aligned_size = roundUpToBlockSize(size);
  int bucket_index = getBucketIndex(aligned_size);
  void* p = nullptr;
  if (free_blocks_[bucket_index].try_dequeue(p)) {
    return p;
  } else {
    cudaError_t err = cudaMalloc(&p, aligned_size);
    CHECK(err == cudaSuccess) << "cudaMalloc failed: " << cudaGetErrorString(err);
  }
  return p;
}

void* PooledAllocStrategy::alloc(size_t size) { return getOrAllocBlock(size); }

void PooledAllocStrategy::free(void* ptr, size_t size) {
  int bucket_index = getBucketIndex(roundUpToBlockSize(size));
  free_blocks_[bucket_index].enqueue(ptr);
}

}  // namespace ginfer::memory::cuda