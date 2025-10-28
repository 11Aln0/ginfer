#pragma once

#define CUDA_CHECK(call)                                                                    \
  do {                                                                                      \
    cudaError_t e = (call);                                                                 \
    if (e != cudaSuccess) {                                                                 \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
      exit(1);                                                                              \
    }                                                                                       \
  } while (0)
  