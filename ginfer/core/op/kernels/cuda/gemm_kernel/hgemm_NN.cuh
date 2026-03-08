#pragma once

#include <cuda_fp16.h>
#include <stdint.h>
#include <stdio.h>
#include "ginfer/core/op/kernels/cuda/intrinsic.cuh"
#include "mma.h"

__device__ __forceinline__ int swizzle_a_offset(int offset) {
  // swizzle smem_a addr to avoid bank conflict
  // m(0-8) is used by mma concurrently, only consider that block(8x16)
  // swizle: B=1, M=3(vector size), S=3(64half=128byte=bank_layer, 64/vector_size=8)

  // cutlass implementation
  int bit_msk = (1 << 1) - 1;        // B=1
  int yyy_msk = bit_msk << (3 + 3);  // M=3, S=3
  int msk_sft = 3;                   // S=3
  return offset ^ ((offset & yyy_msk) >> msk_sft);
}

__device__ __forceinline__ int swizzle_b_offset(int offset) {
  // swizle: B=3, M=3(vector size), S=4(128half=256byte=2bank_layer, 128/vector_size=16)
  // 2 bank_layer are access seperately

  // cutlass implementation
  int bit_msk = (1 << 3) - 1;        // B=3
  int yyy_msk = bit_msk << (3 + 4);  // M=3, S=4
  int msk_sft = 4;                   // S=4
  return offset ^ ((offset & yyy_msk) >> msk_sft);
}

template <typename T, int BYTES = 16>
__device__ __forceinline__ int get_cpasync_src_size(int dim0,
                                                    int dim0_extent,
                                                    int dim1,
                                                    int dim1_extent) {
  int rem_bytes = (dim1_extent - dim1) * sizeof(T);
  int size = min(max(0, rem_bytes), BYTES);
  return (dim0 < dim0_extent) ? size : 0;
}

// bank conflict elimination by swizzle
// stage n pipeline
template <int BM = 128,
          int BN = 128,
          int BK = 16,
          int K_STAGE = 2,
          int MMA_M = 16,
          int MMA_N = 8,
          int MMA_K = 16,
          int BLOCK_TILE_M = 2,
          int BLOCK_TILE_N = 4,
          int WARP_TILE_M = 4,
          int WARP_TILE_N = 4>
__global__ void __launch_bounds__(256) mma2x4_warp4x4_bce_swizzle_stagen_hgemm_kernel(
    const size_t M, const size_t N, const size_t K, const half *A, const half *B, half *C) {
  // attention: N/K must be 16 bytes alignment

  const int bx = blockIdx.x;
  const int by = blockIdx.y;

  static_assert(BM == MMA_M * BLOCK_TILE_M * WARP_TILE_M, "BM mismatch");
  static_assert(BN == MMA_N * BLOCK_TILE_N * WARP_TILE_N, "BN mismatch");
  static_assert(BK == MMA_K, "BK mismatch");

  __shared__ half smem_a[K_STAGE][BM * BK];  // (128, 16)
  __shared__ half smem_b[K_STAGE][BK * BN];  // (16, 128)

  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  const int warp_id = tid / 32;
  const int lane_id = tid % 32;
  // (w0, w2, w4, w6)
  // (w1, w3, w5, w7)
  const int warp_m = warp_id % BLOCK_TILE_M;
  const int warp_n = warp_id / BLOCK_TILE_M;

  // each thread load 8half, totally 128 * 2 = 256 threads
  // logical a layout used by global addr
  int load_smem_logical_a_m = tid / 2;
  int load_smem_logical_a_k = (tid % 2) * 8;

  int load_smem_logical_b_k = tid / (BN / 8);
  int load_smem_logical_b_n = (tid % (BN / 8)) * 8;

  int load_gmem_a_m = by * BM + load_smem_logical_a_m;
  int load_gmem_b_n = bx * BN + load_smem_logical_b_n;

  int load_smem_a_offset = swizzle_a_offset(load_smem_logical_a_m * BK + load_smem_logical_a_k);
  int load_smem_b_offset = swizzle_b_offset(load_smem_logical_b_k * BN + load_smem_logical_b_n);

  // uint32_t reg_c[WARP_TILE_M][WARP_TILE_N][2] = {0};
  float reg_c[WARP_TILE_M][WARP_TILE_N][4] = {0.0f};

// load A/B to smem
#define CP_ASYNC_AB_BUFFER(load_stage, load_gmem_a_m, load_gmem_a_k, load_gmem_b_k, load_gmem_b_n) \
  uint32_t load_smem_a_ptr = __cvta_generic_to_shared(&smem_a[load_stage][load_smem_a_offset]);    \
  int src_size_a = get_cpasync_src_size<half, 16>(load_gmem_a_m, M, load_gmem_a_k, K);             \
  CP_ASYNC_CG_GUARDED(load_smem_a_ptr, &A[load_gmem_a_m * K + load_gmem_a_k], 16, src_size_a);     \
  uint32_t load_smem_b_ptr = __cvta_generic_to_shared(&smem_b[load_stage][load_smem_b_offset]);    \
  int src_size_b = get_cpasync_src_size<half, 16>(load_gmem_b_k, K, load_gmem_b_n, N);             \
  CP_ASYNC_CG_GUARDED(load_smem_b_ptr, &B[load_gmem_b_k * N + load_gmem_b_n], 16, src_size_b);     \
  CP_ASYNC_COMMIT_GROUP();

#define COMPUTE_ON_SMEM(compute_stage)                                                         \
  uint32_t reg_a[WARP_TILE_M][4];                                                              \
  uint32_t reg_b[WARP_TILE_N][2];                                                              \
                                                                                               \
  _Pragma("unroll") for (int wi = 0; wi < WARP_TILE_M; wi++) {                                 \
    int warp_smem_a_m = warp_m * (MMA_M * WARP_TILE_M) + wi * MMA_M;                           \
    int lane_smem_a_m = warp_smem_a_m + lane_id % 16;                                          \
    int lane_smem_a_k = (lane_id / 16) * 8;                                                    \
    int lane_smem_a_offset = swizzle_a_offset(lane_smem_a_m * BK + lane_smem_a_k);             \
    uint32_t lane_smem_a_ptr =                                                                 \
        __cvta_generic_to_shared(&smem_a[compute_stage][lane_smem_a_offset]);                  \
    LDMATRIX_X4_B16(reg_a[wi][0], reg_a[wi][1], reg_a[wi][2], reg_a[wi][3], lane_smem_a_ptr);  \
  }                                                                                            \
                                                                                               \
  _Pragma("unroll") for (int wj = 0; wj < WARP_TILE_N; wj++) {                                 \
    int warp_smem_b_n = warp_n * (MMA_N * WARP_TILE_N) + wj * MMA_N;                           \
    int lane_smem_b_n = warp_smem_b_n;                                                         \
    int lane_smem_b_k = lane_id % 16;                                                          \
    int lane_smem_b_offset = swizzle_b_offset(lane_smem_b_k * BN + lane_smem_b_n);             \
    uint32_t lane_smem_b_ptr =                                                                 \
        __cvta_generic_to_shared(&smem_b[compute_stage][lane_smem_b_offset]);                  \
    LDMATRIX_X2_TRANS_B16(reg_b[wj][0], reg_b[wj][1], lane_smem_b_ptr);                        \
  }                                                                                            \
                                                                                               \
  _Pragma("unroll") for (int wi = 0; wi < WARP_TILE_M; wi++) {                                 \
    for (int wj = 0; wj < WARP_TILE_N; wj++) {                                                 \
      MMA_FP16_ACCFP32(reg_c[wi][wj][0], reg_c[wi][wj][1], reg_c[wi][wj][2], reg_c[wi][wj][3], \
                       reg_a[wi][0], reg_a[wi][1], reg_a[wi][2], reg_a[wi][3], reg_b[wj][0],   \
                       reg_b[wj][1]);                                                          \
    }                                                                                          \
  }

// prefetch K_STAGE-1 stage
#pragma unroll
  for (int k = 0; k < (K_STAGE - 1); k++) {
    int load_gmem_a_k = k * BK + load_smem_logical_a_k;
    int load_gmem_b_k = k * BK + load_smem_logical_b_k;
    CP_ASYNC_AB_BUFFER(k, load_gmem_a_m, load_gmem_a_k, load_gmem_b_k, load_gmem_b_n);
  }

  CP_ASYNC_WAIT_GROUP(K_STAGE - 2);
  __syncthreads();

  int k_tile_iter = (K + BK - 1) / BK;

#pragma unroll
  for (int k = (K_STAGE - 1); k < k_tile_iter; k++) {
    int compute_stage = (k + 1) % K_STAGE;
    int load_stage = k % K_STAGE;

    int load_gmem_a_k = k * BK + load_smem_logical_a_k;
    int load_gmem_b_k = k * BK + load_smem_logical_b_k;

    CP_ASYNC_AB_BUFFER(load_stage, load_gmem_a_m, load_gmem_a_k, load_gmem_b_k, load_gmem_b_n);
    COMPUTE_ON_SMEM(compute_stage);

    CP_ASYNC_WAIT_GROUP(K_STAGE - 2);
    __syncthreads();
  }

  CP_ASYNC_WAIT_GROUP(0);
  __syncthreads();

  // process last K_STAGE-1 stage

#pragma unroll
  for (int k = 0; k < (K_STAGE - 1); k++) {
    int compute_stage = (k_tile_iter - (K_STAGE - 1) + k) % K_STAGE;
    COMPUTE_ON_SMEM(compute_stage);
  }

// store reg
#pragma unroll
  for (int wi = 0; wi < WARP_TILE_M; wi++) {
#pragma unroll
    for (int wj = 0; wj < WARP_TILE_N; wj++) {
      // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#syntax
      int group_id = lane_id / 4;
      int tid_in_group = lane_id % 4;
      int store_lane_c_m0 = group_id;
      int store_lane_c_m1 = group_id + 8;
      int store_lane_c_n = tid_in_group * 2;

      int store_warp_c_m = warp_m * (MMA_M * WARP_TILE_M) + wi * MMA_M;
      int store_warp_c_n = warp_n * (MMA_N * WARP_TILE_N) + wj * MMA_N;

      int store_gmem_c_m0 = by * BM + store_warp_c_m + store_lane_c_m0;
      int store_gmem_c_m1 = by * BM + store_warp_c_m + store_lane_c_m1;
      int store_gmem_c_n = bx * BN + store_warp_c_n + store_lane_c_n;

      int store_gmem_c_addr0 = store_gmem_c_m0 * N + store_gmem_c_n;
      int store_gmem_c_addr1 = store_gmem_c_m1 * N + store_gmem_c_n;

      bool bound0 = (store_gmem_c_m0 < M) && (store_gmem_c_n < N);
      bool bound1 = (store_gmem_c_m1 < M) && (store_gmem_c_n < N);
      half2 acc0 = __float22half2_rn(FLOAT2(reg_c[wi][wj][0]));
      half2 acc1 = __float22half2_rn(FLOAT2(reg_c[wi][wj][2]));
      ST_GLOBAL_PRED_U32(C + store_gmem_c_addr0, acc0, bound0);
      ST_GLOBAL_PRED_U32(C + store_gmem_c_addr1, acc1, bound1);
    }
  }
}