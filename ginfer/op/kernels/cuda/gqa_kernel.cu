#include <cuda_runtime.h>
#include "ginfer/op/kernels/kernel_registry.h"
#include "ginfer/op/kernels/cuda/vectorize.cuh"
#include "ginfer/op/kernels/gqa_kernel.h"

namespace ginfer::op::kernel {
template <int b, typename T, int vec_size = DefaultVecSize<T>::value>
__device__ __forceinline__ void loadQKVTile(const T* __restrict__ p_global,
                                          int block_seq_len,
                                          int head_dim,
                                          int seq_len_stride,
                                          T* p_smem) {

  using AccessT = AlignedVector<T, vec_size>;
  
  int tid = threadIdx.x;
  int thread_per_row = head_dim / vec_size;
  int rows_per_block = blockDim.x / thread_per_row;

  int smem_row = tid / thread_per_row;
  int smem_col = tid % thread_per_row;

  T* sptr = p_smem + smem_row * head_dim + smem_col * vec_size;
  const T* tptr = p_global + (b * blockIdx.x + smem_row) * seq_len_stride + smem_col * vec_size;
  
  AccessT zero_vec;
  for(int r = 0; r < b; r += rows_per_block) {
    if(r + smem_row < block_seq_len) {
      *reinterpret_cast<AccessT*>(sptr) = *reinterpret_cast<const AccessT*>(tptr);
    } else {
      *reinterpret_cast<AccessT*>(sptr) = zero_vec;
    }
    tptr += rows_per_block * seq_len_stride;
    sptr += rows_per_block * head_dim;
  }
}

#define MMA_M 16
#define MMA_N 8
#define MMA_K 16

#define WARP_SIZE 32
#define WARP_TILE_M 4
#define WARP_TILE_N 4
#define BLOCK_TILE_M 2
#define BLOCK_TILE_N 4

#define LDMATRIX_X2_B16(R0, R1, addr) \
    asm volatile(                                 \
        "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n" \
        : "=r"(R0), "=r"(R1)                      \
        : "r"(addr)                               \
    )

#define LDMATRIX_X2_TRANS_B16(R0, R1, addr) \
    asm volatile(                                 \
        "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];\n" \
        : "=r"(R0), "=r"(R1)                      \
        : "r"(addr)                               \
    )

#define LDMATRIX_X4_B16(R0, R1, R2, R3, addr) \
    asm volatile(                                 \
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n" \
        : "=r"(R0), "=r"(R1), "=r"(R2), "=r"(R3)  \
        : "r"(addr)                               \
    )

#define MMA_FP16_ACCFP16(RD0, RD1, RA0, RA1, RA2, RA3, RB0, RB1, RC0, RC1) \
  asm volatile(                                                    \
      "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "         \
      "{%0, %1}, " \
      "{%2, %3, %4, %5}, " \
      "{%6, %7}, " \
      "{%8, %9};\n" \
      : "=r"(RD0), "=r"(RD1) \
      : "r"(RA0), "r"(RA1), "r"(RA2), "r"(RA3), \
        "r"(RB0), "r"(RB1), \
        "r"(RC0), "r"(RC1) \
  )

template <typename T, int br, int bc>
__device__ __forceinline__ void computeSTile(const T* p_Q,
                                             const T* p_K,
                                             T* p_S,
                                             int head_dim) {

  constexpr int BM = BLOCK_TILE_M * WARP_TILE_M * MMA_M; 
  constexpr int BN = BLOCK_TILE_N * WARP_TILE_N * MMA_N;
  
  static_assert(br == BM, "br must equal to BM");
  static_assert(bc == BN, "bc must equal to BN");

  const int warp_id = threadIdx.x / WARP_SIZE;
  const int lane_id = threadIdx.x % WARP_SIZE;

  // (w0, w2, w4, w6)
  // (w1, w3, w5, w7)
  const int warp_m = warp_id % 2;
  const int warp_n = warp_id / 2;

  uint32_t reg_a[WARP_TILE_M][4];
  uint32_t reg_b[WARP_TILE_N][2];
  uint32_t reg_c[WARP_TILE_M][WARP_TILE_N][2];
  
  #pragma unroll
  for(int k = 0; k < head_dim; k += MMA_K) {

    #pragma unroll
    for(int wi = 0; wi < WARP_TILE_M; wi++) {
      int warp_smem_Q_m = warp_m * (WARP_TILE_M * MMA_M) + wi * MMA_M;
      int lane_smem_Q_m = warp_smem_Q_m + lane_id % 16;
      int lane_smem_Q_k = (lane_id / 16) * 8 + k;
      uint32_t lane_smem_Q_addr = __cvta_generic_to_shared(p_Q + lane_smem_Q_m * head_dim + lane_smem_Q_k);
      LDMATRIX_X4_B16(reg_a[wi][0], reg_a[wi][1], reg_a[wi][2], reg_a[wi][3], lane_smem_Q_addr);
    }

    #pragma unroll
    for(int wj = 0; wj < WARP_TILE_N; wj++) {
      // col-major
      int warp_smem_K_n = warp_n * (WARP_TILE_N * MMA_N) + wj * MMA_N;
      int lane_smem_K_n = warp_smem_K_n + lane_id % 8;
      int lane_smem_K_k = lane_id / 8 * 8 + k;
      uint32_t lane_smem_K_addr = __cvta_generic_to_shared(p_K + lane_smem_K_n * head_dim + lane_smem_K_k);
      LDMATRIX_X2_B16(reg_b[wj][0], reg_b[wj][1], lane_smem_K_addr);
    }

    #pragma unroll
    for(int wi = 0; wi < WARP_TILE_M; wi++) {
      #pragma unroll
      for(int wj = 0; wj < WARP_TILE_N; wj++) {
        MMA_FP16_ACCFP16(
          reg_c[wi][wj][0], reg_c[wi][wj][1],
          reg_a[wi][0], reg_a[wi][1], reg_a[wi][2], reg_a[wi][3],
          reg_b[wj][0], reg_b[wj][1],
          reg_c[wi][wj][0], reg_c[wi][wj][1]
        );
      }
    }
    __syncthreads();
  }               
}


// q: [batch, seq_len, num_heads, head_dim]
template <int br, int bc, typename T>
__global__ void gqaKernelImpl(const T* __restrict__ q,
                              const T* __restrict__ k,
                              const T* __restrict__ v,
                              T* __restrict__ output,
                              const int num_heads,
                              const int kv_heads,
                              const int seq_len,
                              const int head_dim) {

  extern __shared__ T smem[];

  int batch_id = blockIdx.x;
  int head_id = blockIdx.y;
  int q_block_id = blockIdx.z;
  int tid = threadIdx.x;
  
  int q_seq_len_stride = num_heads * head_dim;
  int kv_seq_len_stride = kv_heads * head_dim;
  int kv_head_id = head_id / (num_heads / kv_heads);
                                
  int q_offset = batch_id * seq_len * q_seq_len_stride + q_block_id * br * q_seq_len_stride + head_id * head_dim;
  int k_offset = batch_id * seq_len * kv_seq_len_stride + kv_head_id * head_dim;
  int v_offset = k_offset;
  int out_offset = q_offset;

  T* p_q_block_smem = smem;
  T* p_k_block_smem = p_q_block_smem + br * head_dim;
  T* p_v_block_smem = p_k_block_smem + bc * head_dim;
  T* p_o_block_smem = p_v_block_smem + bc * head_dim;
  T* p_s_block_smem = p_o_block_smem + br * head_dim;
  T* p_l_smem = p_s_block_smem + br * bc;
  T* p_m_smem = p_l_smem + br;

  // load Q block
  loadQKVTile<br, T>(q + q_offset, min(br, seq_len - q_block_id * br), head_dim, q_seq_len_stride, p_q_block_smem);

  for(int c = 0; c < seq_len; c += bc) {
    int block_seq_len = min(bc, seq_len - c);
    // load K,V block
    
    loadQKVTile<bc, T>(k + k_offset, block_seq_len, head_dim, kv_seq_len_stride, p_k_block_smem);
    loadQKVTile<bc, T>(v + v_offset, block_seq_len, head_dim, kv_seq_len_stride, p_v_block_smem);
    // compute S = Q * K^T
    computeSTile<T, br, bc>(p_q_block_smem, p_k_block_smem, p_s_block_smem, head_dim);

    k_offset += kv_seq_len_stride * bc;
    v_offset += kv_seq_len_stride * bc;
  }
  
}

// TODO assert head_dim % 64 == 0
// TODO assert bc==br==128, for my life

} // namespace ginfer::op::kernel