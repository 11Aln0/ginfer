#include <cuda_runtime.h>
#include <glog/logging.h>
#include "ginfer/core/op/kernels/cuda/intrinsic.cuh"
#include "ginfer/core/op/kernels/cuda/vectorize.cuh"
#include "ginfer/core/op/kernels/kernel_registry.h"
#include "ginfer/core/op/kernels/kernels.h"

namespace ginfer::core::op::kernel {

template <typename T, int b, int vec_size = DefaultVecSize<T>::value>
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
  const T* tptr = p_global + smem_row * seq_len_stride + smem_col * vec_size;

  AccessT zero_vec;
  for (int r = 0; r < block_seq_len; r += rows_per_block) {
    if (r + smem_row < block_seq_len) {
      *reinterpret_cast<AccessT*>(sptr) = *reinterpret_cast<const AccessT*>(tptr);
    } else {
      *reinterpret_cast<AccessT*>(sptr) = zero_vec;
    }
    tptr += rows_per_block * seq_len_stride;
    sptr += rows_per_block * head_dim;
  }

  // clear smem
  int start_r = (block_seq_len + rows_per_block - 1) / rows_per_block * rows_per_block;
  sptr = p_smem + (start_r + smem_row) * head_dim + smem_col * vec_size;
  for (int r = start_r + smem_row; r < b; r += rows_per_block) {
    *reinterpret_cast<AccessT*>(sptr) = zero_vec;
    sptr += rows_per_block * head_dim;
  }
}

#define MMA_M 16
#define MMA_N 8
#define MMA_K 16

#define BLOCK_TILE_M 2
#define BLOCK_TILE_N 4

// Q[br, head_dim] @ K^T[head_dim, bc] -> S[br, bc]
template <typename T, int br, int bc>
__device__ __forceinline__ void computeSTile(const T* p_Q, const T* p_K, T* p_S, int head_dim) {
  using MmaTraits = MmaTraits<T, float>;

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
  // uint32_t reg_c[WARP_TILE_M][WARP_TILE_N][2] = {0};
  float reg_c[WARP_TILE_M][WARP_TILE_N][4] = {0.0f};

#pragma unroll
  for (int k = 0; k < head_dim; k += MMA_K) {
#pragma unroll
    for (int wi = 0; wi < WARP_TILE_M; wi++) {
      int warp_smem_Q_m = warp_m * (WARP_TILE_M * MMA_M) + wi * MMA_M;
      int lane_smem_Q_m = warp_smem_Q_m + lane_id % 16;
      int lane_smem_Q_k = (lane_id / 16) * 8 + k;
      uint32_t lane_smem_Q_addr =
          __cvta_generic_to_shared(p_Q + lane_smem_Q_m * head_dim + lane_smem_Q_k);
      LDMATRIX_X4_B16(reg_a[wi][0], reg_a[wi][1], reg_a[wi][2], reg_a[wi][3], lane_smem_Q_addr);
    }

#pragma unroll
    for (int wj = 0; wj < WARP_TILE_N; wj++) {
      // col-major
      int warp_smem_K_n = warp_n * (WARP_TILE_N * MMA_N) + wj * MMA_N;
      int lane_smem_K_n = warp_smem_K_n + lane_id % 8;
      int lane_smem_K_k = lane_id / 8 * 8 + k;
      uint32_t lane_smem_K_addr =
          __cvta_generic_to_shared(p_K + lane_smem_K_n * head_dim + lane_smem_K_k);
      LDMATRIX_X2_B16(reg_b[wj][0], reg_b[wj][1], lane_smem_K_addr);
    }

#pragma unroll
    for (int wi = 0; wi < WARP_TILE_M; wi++) {
#pragma unroll
      for (int wj = 0; wj < WARP_TILE_N; wj++) {
        MmaTraits::compute(reg_c[wi][wj][0], reg_c[wi][wj][1], reg_c[wi][wj][2], reg_c[wi][wj][3],
                           reg_a[wi][0], reg_a[wi][1], reg_a[wi][2], reg_a[wi][3], reg_b[wj][0],
                           reg_b[wj][1]);
      }
    }
  }

  using NumericTraits = NumericTraits<T>;

#pragma unroll
  for (int wi = 0; wi < WARP_TILE_M; wi++) {
#pragma unroll
    for (int wj = 0; wj < WARP_TILE_N; wj++) {
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
      auto acc0 = NumericTraits::fromFloat2(FLOAT2(reg_c[wi][wj][0]));
      auto acc1 = NumericTraits::fromFloat2(FLOAT2(reg_c[wi][wj][2]));
      *reinterpret_cast<AccessT*>(&p_S[lane_smem_S_addr0]) = *reinterpret_cast<AccessT*>(&acc0);
      *reinterpret_cast<AccessT*>(&p_S[lane_smem_S_addr1]) = *reinterpret_cast<AccessT*>(&acc1);
    }
  }
}

// S[br, bc] -> P[br, bc];
// r_new_l = r_l * exp(r_m - new_m) + sum(exp(S - new_m))
template <typename T, int br, int bc, int head_dim>
__device__ __forceinline__ void computePTile(T* p_SP,      /** S/P ptr of smem */
                                             const T* p_m, /** old max smem */
                                             T* p_new_m,   /** new max smem */
                                             float* p_l /** sum smem **/,
                                             int q_seq_len,
                                             int kv_seq_len,
                                             int global_r_base,
                                             int global_c_base) {
  using NumericTraits = NumericTraits<T>;
  constexpr int vec_size = DefaultVecSize<T>::value;
  using AccessType = AlignedVector<T, vec_size>;

  constexpr int thread_per_row = bc / vec_size;
  static_assert(thread_per_row <= WARP_SIZE, "thread_per_row must be <= WARP_SIZE");
  static_assert((thread_per_row & (thread_per_row - 1)) == 0, "thread_per_row must be power of 2");

  int tid = threadIdx.x;
  int row_per_iter = blockDim.x / thread_per_row;

  int smem_row_offset = tid / thread_per_row;
  int smem_col = tid % thread_per_row * vec_size;

  int casual_mask_offset = kv_seq_len - q_seq_len;

  int bound = min(br, q_seq_len - global_r_base);
  for (int r = 0; r < bound; r += row_per_iter) {
    T* sptr = p_SP + (smem_row_offset + r) * bc + smem_col;
    AccessType vec = *reinterpret_cast<const AccessType*>(sptr);

    int global_r = global_r_base + smem_row_offset + r;
    T rsqrt_head_dim = NumericTraits::fromFloat(rsqrtf(static_cast<float>(head_dim)));
#pragma unroll
    for (int j = 0; j < vec_size; j++) {
      int global_c = global_c_base + smem_col + j;
      bool is_valid = global_c <= (global_r + casual_mask_offset) &&
                      global_c < kv_seq_len;  // casual mask and seq_len mask
      vec.val[j] = is_valid ? vec.val[j] * rsqrt_head_dim
                            : NumericTraits::fromFloat(-INFINITY);  // div sqrt(head_dim)
    }

    float old_max = NumericTraits::toFloat(p_m[smem_row_offset + r]);
    float partial_max = old_max;

#pragma unroll
    for (int j = 0; j < vec_size; j++) {
      partial_max = max(partial_max, NumericTraits::toFloat(vec.val[j]));
    }

    float partial_sum = 0.0f;
#pragma unroll
    for (int j = 0; j < vec_size; j++) {
      float val = NumericTraits::toFloat(vec.val[j]);
      float diff = __isinf(vec.val[j]) && __isinf(partial_max) ? -INFINITY : val - partial_max;
      partial_sum += expf(diff);
    }

#pragma unroll
    for (int mask = thread_per_row >> 1; mask >= 1; mask >>= 1) {
      float new_max = max(__shfl_xor_sync(0xFFFFFFFF, partial_max, mask), partial_max);
      float diff = __isinf(partial_max) && __isinf(new_max) ? -INFINITY : partial_max - new_max;
      partial_sum = partial_sum * expf(diff);
      partial_sum += __shfl_xor_sync(0xFFFFFFFF, partial_sum, mask);
      partial_max = new_max;
    }

// re-scale P
#pragma unroll
    for (int j = 0; j < vec_size; j++) {
      vec.val[j] = expf(NumericTraits::toFloat(vec.val[j]) - partial_max);
    }

    // write back P
    *reinterpret_cast<AccessType*>(sptr) = vec;
    if ((tid % thread_per_row) == 0) {
      p_new_m[smem_row_offset + r] = NumericTraits::fromFloat(partial_max);
      p_l[smem_row_offset + r] =
          p_l[smem_row_offset + r] * expf(old_max - partial_max) + partial_sum;
    }
  }
}

// P[br, bc] @ V[bc, head_dim] -> O[br, head_dim]
template <typename T, int br, int bc, int head_dim>
__device__ void updateOTile(
    const T* p_P, const T* p_V, const T* p_old_m, const T* p_new_m, T* p_O) {
  using MmaTraits = MmaTraits<T, float>;
  using NumericTraits = NumericTraits<T>;

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
  // uint32_t reg_c[WARP_TILE_M][WARP_TILE_N][2] = {0};
  float reg_c[WARP_TILE_M][WARP_TILE_N][4] = {0.0f};

#pragma unroll
  for (int k = 0; k < bc; k += MMA_K) {
#pragma unroll
    for (int wi = 0; wi < WARP_TILE_M; wi++) {
      int warp_smem_P_m = warp_m * (WARP_TILE_M * MMA_M) + wi * MMA_M;
      int lane_smem_P_m = warp_smem_P_m + lane_id % 16;
      int lane_smem_P_k = (lane_id / 16) * 8 + k;
      uint32_t lane_smem_P_addr =
          __cvta_generic_to_shared(p_P + lane_smem_P_m * bc + lane_smem_P_k);
      LDMATRIX_X4_B16(reg_a[wi][0], reg_a[wi][1], reg_a[wi][2], reg_a[wi][3], lane_smem_P_addr);
    }

#pragma unroll
    for (int wj = 0; wj < WARP_TILE_N; wj++) {
      // col-major
      int warp_smem_V_n = warp_n * (WARP_TILE_N * MMA_N) + wj * MMA_N;
      int lane_smem_V_n = warp_smem_V_n;
      int lane_smem_V_k = lane_id % 16 + k;
      uint32_t lane_smem_V_addr =
          __cvta_generic_to_shared(p_V + lane_smem_V_k * head_dim + lane_smem_V_n);
      LDMATRIX_X2_TRANS_B16(reg_b[wj][0], reg_b[wj][1], lane_smem_V_addr);
    }

#pragma unroll
    for (int wi = 0; wi < WARP_TILE_M; wi++) {
#pragma unroll
      for (int wj = 0; wj < WARP_TILE_N; wj++) {
        MmaTraits::compute(reg_c[wi][wj][0], reg_c[wi][wj][1], reg_c[wi][wj][2], reg_c[wi][wj][3],
                           reg_a[wi][0], reg_a[wi][1], reg_a[wi][2], reg_a[wi][3], reg_b[wj][0],
                           reg_b[wj][1]);
      }
    }
  }
#pragma unroll
  for (int wi = 0; wi < WARP_TILE_M; wi++) {
#pragma unroll
    for (int wj = 0; wj < WARP_TILE_N; wj++) {
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
      old_O.val[0] = NumericTraits::toFloat(old_O.val[0]) * scale0;
      old_O.val[1] = NumericTraits::toFloat(old_O.val[1]) * scale0;
      auto acc0 = NumericTraits::fromFloat2(FLOAT2(reg_c[wi][wj][0]));
      old_O = old_O + *reinterpret_cast<AccessT*>(&acc0);
      *reinterpret_cast<AccessT*>(&p_O[lane_smem_S_addr0]) = old_O;

      old_O = *reinterpret_cast<AccessT*>(&p_O[lane_smem_S_addr1]);
      old_O.val[0] = NumericTraits::toFloat(old_O.val[0]) * scale1;
      old_O.val[1] = NumericTraits::toFloat(old_O.val[1]) * scale1;
      auto acc1 = NumericTraits::fromFloat2(FLOAT2(reg_c[wi][wj][2]));
      old_O = old_O + *reinterpret_cast<AccessT*>(&acc1);
      *reinterpret_cast<AccessT*>(&p_O[lane_smem_S_addr1]) = old_O;
    }
  }
}

template <typename T, int head_dim>
__device__ void storeOTile(
    const T* p_o_block_smem, T* p_output, float* p_l, int block_seq_len, int seq_len_stride) {
  using NumericTraits = NumericTraits<T>;
  constexpr int vec_size = DefaultVecSize<T>::value;
  using AccessType = AlignedVector<T, vec_size>;

  int tid = threadIdx.x;
  int thread_per_row = head_dim / vec_size;
  int row_per_iter = blockDim.x / thread_per_row;

  int start_row = tid / thread_per_row;
  int col = tid % thread_per_row * vec_size;

#pragma unroll
  for (int row = start_row; row < block_seq_len; row += row_per_iter) {
    const T* sptr = p_o_block_smem + row * head_dim + col;
    T* gptr = p_output + row * seq_len_stride + col;

    auto vec = *reinterpret_cast<const AccessType*>(sptr);
    float inv_l = __fdividef(1.0f, p_l[row]);
#pragma unroll
    for (int i = 0; i < vec_size; i++) {
      vec.val[i] = NumericTraits::toFloat(vec.val[i]) * inv_l;
    }

    *reinterpret_cast<AccessType*>(gptr) = vec;
  }
}

template <typename T, int br, int head_dim>
__device__ void initOTile(T* p_o_block_smem) {
  constexpr int vec_size = DefaultVecSize<T>::value;
  using AccessType = AlignedVector<T, vec_size>;

  int tid = threadIdx.x;
  int thread_per_row = head_dim / vec_size;
  int row_per_iter = blockDim.x / thread_per_row;

  int start_row = tid / thread_per_row;
  int col = tid % thread_per_row * vec_size;

  AccessType zero_vec;
#pragma unroll
  for (int row = start_row; row < br; row += row_per_iter) {
    T* sptr = p_o_block_smem + row * head_dim + col;
    *reinterpret_cast<AccessType*>(sptr) = zero_vec;
  }
}

template <typename T, int br>
__device__ void initReduceVal(T* p_m0_smem, T* p_m1_smem, float* p_l_smem) {
  if (threadIdx.x < br) {
    p_m0_smem[threadIdx.x] = -INFINITY;
    p_m1_smem[threadIdx.x] = -INFINITY;
    p_l_smem[threadIdx.x] = 0.0f;
  }
}

// q: [batch, seq_len, num_heads, head_dim] padded to [batch, context_size, num_heads, head_dim]
template <typename T, int br, int bc, int head_dim>
__global__ void GQAKernelImpl(const T* __restrict__ q,
                              const T* __restrict__ k,
                              const T* __restrict__ v,
                              T* __restrict__ output,
                              const int num_heads,
                              const int kv_heads,
                              const int q_seq_len,
                              const int kv_seq_len,
                              size_t q_batch_stride,
                              size_t kv_batch_stride) {
  static_assert(head_dim % 64 == 0, "head_dim must be multiple of 64");

  extern __shared__ char smem[];

  int batch_id = blockIdx.x;
  int head_id = blockIdx.y;
  int q_block_id = blockIdx.z;

  int q_seq_len_stride = num_heads * head_dim;
  int kv_seq_len_stride = kv_heads * head_dim;
  int out_seq_len_stride = q_seq_len_stride;

  int kv_head_id = head_id / (num_heads / kv_heads);

  int q_offset =
      batch_id * q_batch_stride + q_block_id * br * q_seq_len_stride + head_id * head_dim;
  int k_offset = batch_id * kv_batch_stride + kv_head_id * head_dim;
  int v_offset = k_offset;
  int out_offset = q_offset;

  T* p_q_block_smem = reinterpret_cast<T*>(smem);
  T* p_k_block_smem = p_q_block_smem + br * head_dim;
  T* p_v_block_smem = p_k_block_smem + bc * head_dim;
  T* p_o_block_smem = p_v_block_smem + bc * head_dim;
  T* p_s_block_smem = p_o_block_smem + br * head_dim;
  T* p_m_smem = p_s_block_smem + br * bc;  // 2 x br, one for old max, one for new max
  float* p_l_smem = reinterpret_cast<float*>(p_m_smem + 2 * br);

  initOTile<T, br, head_dim>(p_o_block_smem);
  initReduceVal<T, br>(p_m_smem, p_m_smem + br, p_l_smem);
  __syncthreads();

  // load Q block
  loadQKVTile<T, br>(q + q_offset, min(br, q_seq_len - q_block_id * br), head_dim, q_seq_len_stride,
                     p_q_block_smem);

  for (int c = 0; c < kv_seq_len; c += bc) {
    if (c >= (q_block_id + 1) * br + kv_seq_len - q_seq_len) break;  // causal
    int kv_block_seq_len = min(bc, kv_seq_len - c);
    // load K,V block

    loadQKVTile<T, bc>(k + k_offset, kv_block_seq_len, head_dim, kv_seq_len_stride, p_k_block_smem);
    loadQKVTile<T, bc>(v + v_offset, kv_block_seq_len, head_dim, kv_seq_len_stride, p_v_block_smem);
    __syncthreads();
    // compute S = Q * K^T
    computeSTile<T, br, bc>(p_q_block_smem, p_k_block_smem, p_s_block_smem, head_dim);
    __syncthreads();
    T* p_old_m = p_m_smem + ((c / bc) & 1) * br;
    T* p_new_m = p_m_smem + (((c / bc) & 1) ^ 1) * br;
    int global_r_base = q_block_id * br, global_c_base = c;  // for causal
    computePTile<T, br, bc, head_dim>(p_s_block_smem, p_old_m, p_new_m, p_l_smem, q_seq_len,
                                      kv_seq_len, global_r_base, global_c_base);
    __syncthreads();
    // compute O += P * V
    updateOTile<T, br, bc, head_dim>(p_s_block_smem, p_v_block_smem, p_old_m, p_new_m,
                                     p_o_block_smem);
    __syncthreads();

    k_offset += kv_seq_len_stride * bc;
    v_offset += kv_seq_len_stride * bc;
  }

  storeOTile<T, head_dim>(p_o_block_smem, output + out_offset, p_l_smem,
                          min(br, q_seq_len - q_block_id * br), out_seq_len_stride);

  // __syncthreads();
  // if(threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
  //   for(int i = 0; i < 16; i++) {
  //     printf("%d: q: %f, k: %f, v: %f, o: %f\n", i, static_cast<float>(q[i]),
  //     static_cast<float>(k[i]), static_cast<float>(v[i]), static_cast<float>(output[i]));
  //   }
  // }
  // store O block
}

// q: [q_seq_len, num_heads, head_dim]
// k, v: [kv_seq_len, num_heads_kv, head_dim] where num_heads_kv divides num_heads
template <typename T, typename Context>
void GQAKernel(const Context& ctx,
               const tensor::Tensor& q,
               const tensor::Tensor& k,
               const tensor::Tensor& v,
               tensor::Tensor& output) {
  CHECK(ctx.getDeviceType() == common::DeviceType::kDeviceCUDA)
      << "GQAKernel only supports CUDA device type.";
  auto cuda_ctx = static_cast<const common::CUDADeviceContext&>(ctx);

  const auto& q_shape = q.shape();
  const auto& q_strides = q.strides();
  const int batch = 1;
  const int q_seq_len = q_shape[0];
  const int num_heads = q_shape[1];
  const int head_dim = q_shape[2];
  const size_t q_batch_stride = 0;

  const auto& k_shape = k.shape();
  const auto& k_strides = k.strides();
  const int kv_seq_len = k_shape[0];
  const int kv_heads = k_shape[1];
  const size_t kv_batch_stride = 0;

  const T* p_q = q.data<T>();
  const T* p_k = k.data<T>();
  const T* p_v = v.data<T>();
  T* p_output = output.data<T>();

  constexpr int br = 64, bc = 64;

  const int block_dim = 256;
  dim3 grid_dim(batch, num_heads, (q_seq_len + br - 1) / br);
  size_t smem_size = (br * head_dim * 2 +  // Q, O
                      bc * head_dim * 2 +  // K, V
                      br * bc +            // S/P
                      2 * br) *
                         sizeof(T) +
                     br * sizeof(float);

  auto dispatcher = [&]() {
    DLOG(INFO) << "GQA kernel smem size: " << smem_size;
    switch (head_dim) {
      case 64:
        cudaFuncSetAttribute(GQAKernelImpl<T, br, bc, 64>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
        GQAKernelImpl<T, br, bc, 64><<<grid_dim, block_dim, smem_size, cuda_ctx.getStream()>>>(
            p_q, p_k, p_v, p_output, num_heads, kv_heads, q_seq_len, kv_seq_len, q_batch_stride,
            kv_batch_stride);
        break;
      case 128:
        cudaFuncSetAttribute(GQAKernelImpl<T, br, bc, 128>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
        GQAKernelImpl<T, br, bc, 128><<<grid_dim, block_dim, smem_size, cuda_ctx.getStream()>>>(
            p_q, p_k, p_v, p_output, num_heads, kv_heads, q_seq_len, kv_seq_len, q_batch_stride,
            kv_batch_stride);
        break;
      default:
        throw std::runtime_error("Unsupported head_dim in gqaKernel dispatcher");
    }
  };

  dispatcher();
}

REGISTER_KERNEL(GQA, CUDA, GQAKernel, Float16, BFloat16);

}  // namespace ginfer::core::op::kernel