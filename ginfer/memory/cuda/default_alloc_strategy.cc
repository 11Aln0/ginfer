#include <cuda_runtime.h>
#include <glog/logging.h>
#include "ginfer/memory/cuda/alloc_strategy.h"

namespace ginfer::memory::cuda {

void* DefaultAllocStrategy::alloc(size_t size) {
  if (size == 0) {
    LOG(WARNING) << "Try to allocate 0 bytes.";
    return nullptr;
  }
  void* ptr = nullptr;
  cudaError_t err = cudaMalloc(&ptr, size);
  CHECK_EQ(err, cudaSuccess) << "cudaMalloc failed: " << cudaGetErrorString(err);
  return ptr;
}

void DefaultAllocStrategy::free(void* ptr, size_t size) {
  (void)size;
  if (!ptr) {
    LOG(WARNING) << "Try to free a nullptr.";
    return;
  }
  cudaError_t err = cudaFree(ptr);
  CHECK_EQ(err, cudaSuccess) << "cudaFree failed: " << cudaGetErrorString(err);
}

}  // namespace ginfer::memory::cuda