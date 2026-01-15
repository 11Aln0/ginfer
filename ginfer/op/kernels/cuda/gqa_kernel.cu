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

// Q[br, head_dim] @ K^T[head_dim, bc] -> S[br, bc]
template <typename T, int br, int bc>
__device__ __forceinline__ void computeSTile(const T* p_Q,
                                             const T* p_K,
                                             T* p_S,
                                             int head_dim) {

  constexpr int WARP_TILE_M = br / (BLOCK_TILE_M * MMA_M);
  constexpr int WARP_TILE_N = bc / (BLOCK_TILE_N * MMA_N);

  const int warp_id = threadIdx.x / WARP_SIZE;
  const int lane_id = threadIdx.x % WARP_SIZE;

  // (w0, w2, w4, w6)
  // (w1, w3, w5, w7)
  const int warp_m = warp_id % 2;
  const int warp_n = warp_id / 2;

  uint32_t reg_a[WARP_TILE_M][4];
  uint32_t reg_b[WARP_TILE_N][2];
  uint32_t reg_c[WARP_TILE_M][WARP_TILE_N][2] = {0};
  
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
  }
  #pragma unroll
  for(int wi = 0; wi < WARP_TILE_M; wi++) {
    #pragma unroll
    for(int wj = 0; wj < WARP_TILE_N; wj++) {
      int group_id = lane_id / 4;
      int tid_in_group = lane_id % 4;

      int warp_smem_S_m = warp_m * (WARP_TILE_M * MMA_M) + wi * MMA_M;
      int warp_smem_S_n = warp_n * (WARP_TILE_N * MMA_N) + wj * MMA_N;

      int lane_smem_S_m0 = warp_smem_S_m + group_id;
      int lane_smem_S_m1 = lane_smem_S_m0 + 8;
      int lane_smem_S_n = warp_smem_S_n + tid_in_group * 2;

      int lane_smem_S_addr0 = lane_smem_S_m0 * bc + lane_smem_S_n;
      int lane_smem_S_addr1 = lane_smem_S_m1 * bc + lane_smem_S_n;

      using AccessT = AlignedVector<T, 2>;
      *reinterpret_cast<AccessT*>(&p_S[lane_smem_S_addr0]) =
          *reinterpret_cast<AccessT*>(&reg_c[wi][wj][0]);
      *reinterpret_cast<AccessT*>(&p_S[lane_smem_S_addr1]) =
          *reinterpret_cast<AccessT*>(&reg_c[wi][wj][1]);
    }
  }          
}

// S[br, bc] -> P[br, bc]; 
// r_new_l = r_l * exp(r_m - new_m) + sum(exp(S - new_m))
template <typename T, int br, int bc>
__device__ __forceinline__ void computePTile(T* p_SP, /** S/P ptr of smem */
                                             const T* p_m, /** old max smem */
                                             T* p_new_m, /** new max smem */
                                             float* r_l /** sum reg */
                                            ) {

  int tid = threadIdx.x;
  int thread_per_row = blockDim.x / br;
  int thread_stride = bc / thread_per_row;
  
  int smem_row = tid / thread_per_row;
  int smem_col = tid % thread_per_row * thread_stride;

  T* sptr = p_SP + smem_row * bc + smem_col;
  
  constexpr int vec_size = DefaultVecSize<T>::value;
  using AccessType = AlignedVector<T, vec_size>;
  
  T old_max = p_m[smem_row];
  T new_max = old_max;
  for(int i = 0; i < thread_stride; i += vec_size) {
    AccessType vec = *reinterpret_cast<const AccessType*>(sptr); // TODO bank conflict
    #pragma unroll
    for(int j = 0; j < vec_size; j++) {
      new_max = max(new_max, vec.val[j]);
    }
    sptr += vec_size;
  }
  
  int lane_id = tid % 32;
  // reduce within row
  T tmp = __shfl_sync(0xFFFFFFFF, new_max, lane_id ^ 1);
  new_max = max(tmp, new_max);

  float partial_sum = 0.0f;
  sptr = p_SP + smem_row * bc + smem_col;
  for(int i = 0; i < thread_stride; i += vec_size) {
    AccessType vec = *reinterpret_cast<const AccessType*>(sptr); 
    #pragma unroll
    for(int j = 0; j < vec_size; j++) {
      vec.val[j] = expf(vec.val[j] - new_max);
      partial_sum += vec.val[j];
    }
    *reinterpret_cast<AccessType*>(sptr) = vec;
    sptr += vec_size;
  }

  partial_sum += __shfl_sync(0xFFFFFFFF, partial_sum, lane_id ^ 1);
  
  // update row new sum
  *r_l = *r_l * expf(old_max - new_max) + partial_sum;
  p_new_m[smem_row] = new_max;
}

// P[br, bc] @ V[bc, head_dim] -> O[br, head_dim]
template <typename T, int br, int bc, int head_dim>
__global__ void updateOTile(const T* p_P,
                            const T* p_V,
                            const T* p_old_m,
                            const T* p_new_m,
                            T *p_O) {
                        
  constexpr int WARP_TILE_M = br / (BLOCK_TILE_M * MMA_M);
  constexpr int WARP_TILE_N = head_dim / (BLOCK_TILE_N * MMA_N);

  const int warp_id = threadIdx.x / WARP_SIZE;
  const int lane_id = threadIdx.x % WARP_SIZE;

  // (w0, w2, w4, w6)
  // (w1, w3, w5, w7)
  const int warp_m = warp_id % 2;
  const int warp_n = warp_id / 2;

  uint32_t reg_a[WARP_TILE_M][4];
  uint32_t reg_b[WARP_TILE_N][2];
  uint32_t reg_c[WARP_TILE_M][WARP_TILE_N][2] = {0};
  
  #pragma unroll
  for(int k = 0; k < bc; k += MMA_K) {

    #pragma unroll
    for(int wi = 0; wi < WARP_TILE_M; wi++) {
      int warp_smem_P_m = warp_m * (WARP_TILE_M * MMA_M) + wi * MMA_M;
      int lane_smem_P_m = warp_smem_P_m + lane_id % 16;
      int lane_smem_P_k = (lane_id / 16) * 8 + k;
      uint32_t lane_smem_P_addr = __cvta_generic_to_shared(p_P + lane_smem_P_m * head_dim + lane_smem_P_k);
      LDMATRIX_X4_B16(reg_a[wi][0], reg_a[wi][1], reg_a[wi][2], reg_a[wi][3], lane_smem_P_addr);
    }

    #pragma unroll
    for(int wj = 0; wj < WARP_TILE_N; wj++) {
      // col-major
      int warp_smem_V_n = warp_n * (WARP_TILE_N * MMA_N) + wj * MMA_N;
      int lane_smem_V_n = warp_smem_V_n;
      int lane_smem_V_k = lane_id % 16 + k;
      uint32_t lane_smem_V_addr = __cvta_generic_to_shared(p_V + lane_smem_V_n * head_dim + lane_smem_V_k);
      LDMATRIX_X2_TRANS_B16(reg_b[wj][0], reg_b[wj][1], lane_smem_V_addr);
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
  }
  #pragma unroll
  for(int wi = 0; wi < WARP_TILE_M; wi++) {
    #pragma unroll
    for(int wj = 0; wj < WARP_TILE_N; wj++) {
      int group_id = lane_id / 4;
      int tid_in_group = lane_id % 4;

      int warp_smem_S_m = warp_m * (WARP_TILE_M * MMA_M) + wi * MMA_M;
      int warp_smem_S_n = warp_n * (WARP_TILE_N * MMA_N) + wj * MMA_N;

      int lane_smem_S_m0 = warp_smem_S_m + group_id;
      int lane_smem_S_m1 = lane_smem_S_m0 + 8;
      int lane_smem_S_n = warp_smem_S_n + tid_in_group * 2;

      int lane_smem_S_addr0 = lane_smem_S_m0 * head_dim + lane_smem_S_n;
      int lane_smem_S_addr1 = lane_smem_S_m1 * head_dim + lane_smem_S_n;

      using AccessT = AlignedVector<T, 2>;
      float old_max0 = p_old_m[lane_smem_S_m0];
      float new_max0 = p_new_m[lane_smem_S_m0];
      float scale0 = expf(old_max0 - new_max0);

      float old_max1 = p_old_m[lane_smem_S_m1];
      float new_max1 = p_new_m[lane_smem_S_m1];
      float scale1 = expf(old_max1 - new_max1);

      auto old_O = *reinterpret_cast<AccessT*>(&p_O[lane_smem_S_addr0]);
      old_O.val[0] = old_O.val[0] * scale0;
      old_O.val[1] = old_O.val[1] * scale0; 
      old_O = old_O + *reinterpret_cast<AccessT*>(&reg_c[wi][wj][0]);
      *reinterpret_cast<AccessT*>(&p_O[lane_smem_S_addr0]) = old_O;
      
      old_O = *reinterpret_cast<AccessT*>(&p_O[lane_smem_S_addr1]);
      old_O.val[0] = old_O.val[0] * scale1;
      old_O.val[1] = old_O.val[1] * scale1; 
      old_O = old_O + *reinterpret_cast<AccessT*>(&reg_c[wi][wj][1]);
      *reinterpret_cast<AccessT*>(&p_O[lane_smem_S_addr1]) = old_O;
    }
  }  
}


template<typename T, int br, int bc, int head_dim>
__device__ void storeOTile(const T* p_o_block_smem,
                           float reg_l,
                           T* p_output) {

  constexpr int vec_size = DefaultVecSize<T>::value;
  using AccessType = AlignedVector<T, vec_size>;

  int tid = threadIdx.x;
  int thread_per_row = blockDim.x / br;
  int thread_stride = head_dim / thread_per_row;
  
  int smem_row = tid / thread_per_row;
  int smem_col = tid % thread_per_row * thread_stride;
}


// q: [batch, seq_len, num_heads, head_dim]
template <typename T, int br, int bc, int head_dim>
__global__ void gqaKernelImpl(const T* __restrict__ q,
                              const T* __restrict__ k,
                              const T* __restrict__ v,
                              T* __restrict__ output,
                              const int num_heads,
                              const int kv_heads,
                              const int seq_len) {

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
  
  // TODO init O, max
  T* p_q_block_smem = smem;
  T* p_k_block_smem = p_q_block_smem + br * head_dim;
  T* p_v_block_smem = p_k_block_smem + bc * head_dim;
  T* p_o_block_smem = p_v_block_smem + bc * head_dim;
  T* p_s_block_smem = p_o_block_smem + br * head_dim;
  T* p_m_smem = p_s_block_smem + br * bc; // 2 x br, one for old max, one for new max
  T* p_l_smem = p_m_smem + 2 * br; 

  float reg_l = 0.0f; // 128x128 S, 256 thread, each two thread has same value

  // load Q block
  loadQKVTile<br, T>(q + q_offset, min(br, seq_len - q_block_id * br), head_dim, q_seq_len_stride, p_q_block_smem);

  for(int c = 0; c < seq_len; c += bc) {
    int block_seq_len = min(bc, seq_len - c);
    // load K,V block
    
    loadQKVTile<bc, T>(k + k_offset, block_seq_len, head_dim, kv_seq_len_stride, p_k_block_smem);
    loadQKVTile<bc, T>(v + v_offset, block_seq_len, head_dim, kv_seq_len_stride, p_v_block_smem);
    __syncthreads();
    // compute S = Q * K^T
    computeSTile<T, br, bc>(p_q_block_smem, p_k_block_smem, p_s_block_smem, head_dim);
    __syncthreads();
    T* p_old_m = p_m_smem + (c & 1) * br;
    T* p_new_m = p_m_smem + ((c & 1) ^ 1) * br;
    computePTile<T, br, bc>(p_s_block_smem, p_old_m, p_new_m, &reg_l);
    __syncthreads();
    // compute O += P * V
    updateOTile<T, br, bc, head_dim>(p_s_block_smem, p_v_block_smem, p_old_m, p_new_m, p_o_block_smem);
    __syncthreads();

    k_offset += kv_seq_len_stride * bc;
    v_offset += kv_seq_len_stride * bc;
  }
  
  // store O block
  storeOTile<T, br, bc, head_dim>(p_o_block_smem, reg_l, output + out_offset);
}

// TODO assert head_dim % 64 == 0
// TODO assert bc==br==128, for my life
// TODO assert threadDim == 256

} // namespace ginfer::op::kernel