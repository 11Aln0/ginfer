#include <cuda_runtime.h>
#include <glog/logging.h>
#include <cub/block/block_reduce.cuh>
#include "ginfer/core/op/kernels/cuda/intrinsic.cuh"
#include "ginfer/core/op/kernels/cuda/vectorize.cuh"
#include "ginfer/core/op/kernels/kernel_registry.h"
#include "ginfer/core/op/kernels/page_attn_kernel.h"

namespace ginfer::core::op::kernel {

template <typename T, int head_dim>
__device__ __forceinline__ void loadQHead(const T* __restrict__ p_global, T* p_smem) {
  int tid = threadIdx.x;

  if (tid < head_dim) {
    p_smem[tid] = p_global[tid];
  }
}

template <typename T, int head_dim, int vec_size = DefaultVecSize<T>::value>
__device__ __forceinline__ void loadKVPagedTileAsync(const T* __restrict__ p_global,
                                                     T* p_smem,
                                                     const int* __restrict__ block_table,
                                                     int seq_base,
                                                     int kv_seqlen,
                                                     int seqlen_stride,
                                                     int paged_block_size) {
  static_assert(head_dim % vec_size == 0, "head_dim must be divisible by vec_size");

  int tid = threadIdx.x;
  int thread_per_seq = head_dim / vec_size;

  int smem_row = tid / thread_per_seq;
  int smem_col = tid % thread_per_seq;

  T* sptr_base = p_smem + smem_col * vec_size;
  const T* gptr_base = p_global + smem_col * vec_size;

  int logical_seq = seq_base + smem_row;
  int btbl_idx =
      logical_seq / paged_block_size;  // TODO bit operator if paged_block_size is power of 2
  int block_id = logical_seq < kv_seqlen ? block_table[btbl_idx] : 0;
  int phy_seq = block_id * paged_block_size + logical_seq % paged_block_size;

  T* sptr = sptr_base + smem_row * head_dim;
  const T* gptr = gptr_base + phy_seq * seqlen_stride;

  int cp_async_bytes = logical_seq < kv_seqlen ? 16 : 0;
  uint32_t smem_addr = __cvta_generic_to_shared(sptr);
  CP_ASYNC_CG_GUARDED(smem_addr, gptr, 16, cp_async_bytes);
  CP_ASYNC_COMMIT_GROUP();
}

// Q[1, head_dim] @ K^T[head_dim, bc] -> S[1, bc]
template <typename T, int head_dim>
__device__ __forceinline__ void computeSTileStage(const T* p_Q,
                                                  const T* p_K,  // [kv_stage_rows, head_dim]
                                                  T* p_S,        // [bc]
                                                  int kv_stage_rows) {
  constexpr int vec_size = head_dim / WARP_SIZE;
  using AccessT = AlignedVector<T, vec_size>;

  int tid = threadIdx.x;
  int lane_id = tid % WARP_SIZE;

  int smem_row = tid / WARP_SIZE;
  int smem_col = lane_id * vec_size;

  const AccessT* p_Q_vec = reinterpret_cast<const AccessT*>(p_Q + smem_col);

  for (int i = smem_row; i < kv_stage_rows; i += blockDim.x / WARP_SIZE) {
    const AccessT* p_K_vec = reinterpret_cast<const AccessT*>(p_K + i * head_dim + smem_col);

    float acc = DotProduct<T, vec_size>::run(*p_Q_vec, *p_K_vec);

    // reduction within warp
    for (int offset = WARP_SIZE >> 1; offset > 0; offset >>= 1) {
      acc += __shfl_down_sync(0xffffffff, acc, offset);
    }

    if (lane_id == 0) {
      p_S[i] = static_cast<T>(acc * rsqrtf(static_cast<float>(head_dim)));
    }
  }
}

template <typename T, int bc, int head_dim>
__device__ __forceinline__ void computeSTile(const T* p_Q_smem,
                                             const T* p_K_gmem,
                                             T* p_K_smem,
                                             T* p_S_smem,
                                             const int* block_table,
                                             int seq_base,
                                             int kv_seqlen,
                                             int seqlen_stride,
                                             int paged_block_size,
                                             int kv_stage_rows) {
  loadKVPagedTileAsync<T, head_dim>(p_K_gmem, p_K_smem, block_table, seq_base, kv_seqlen,
                                    seqlen_stride, paged_block_size);
  CP_ASYNC_WAIT_ALL();
  __syncthreads();

  int seqlen = min(kv_seqlen - seq_base, bc);
  int stages = (seqlen + kv_stage_rows - 1) / kv_stage_rows;
  for (int i = 0; i < stages; i++) {
    int smem_offset_K = (i % 2) * kv_stage_rows * head_dim;
    int smem_offset_S = i * kv_stage_rows;
    int next_smem_offset_K = ((i + 1) % 2) * kv_stage_rows * head_dim;
    if (i + 1 < stages) {
      loadKVPagedTileAsync<T, head_dim>(p_K_gmem, p_K_smem + next_smem_offset_K, block_table,
                                        seq_base + (i + 1) * kv_stage_rows, kv_seqlen,
                                        seqlen_stride, paged_block_size);
    }

    int valid_rows = min(kv_stage_rows, seqlen - i * kv_stage_rows);
    computeSTileStage<T, head_dim>(p_Q_smem, p_K_smem + smem_offset_K, p_S_smem + smem_offset_S,
                                   valid_rows);
    CP_ASYNC_WAIT_ALL();
    __syncthreads();
  }
}

// S[1, bc] -> P[1, bc];
template <typename T, int bc, int head_dim, int thread_per_block>
__device__ __forceinline__ void computePTile(T* p_SP,        /** S/P ptr of smem */
                                             const T& old_m, /** old max reg*/
                                             T& new_m,       /** new max reg */
                                             float& l,       /** sum reg **/
                                             int kv_seqlen,
                                             int global_c_base) {
  using Traits = NumericTraits<T>;
  using MaxBlockReduce = cub::BlockReduce<float, thread_per_block>;
  using SumBlockReduce = cub::BlockReduce<float, thread_per_block>;
  constexpr int vec_size = DefaultVecSize<T>::value;
  using AccessT = AlignedVector<T, vec_size>;

  static_assert(bc % (thread_per_block * vec_size) == 0,
                "bc must be divisible by thread_per_block * vec_size");

  __shared__ typename MaxBlockReduce::TempStorage max_temp_storage;
  __shared__ typename SumBlockReduce::TempStorage sum_temp_storage;
  __shared__ float smem_new_m;
  __shared__ float smem_l;

  const int tid = threadIdx.x;
  float partial_max = old_m;

  for (int idx = tid * vec_size; idx < bc; idx += thread_per_block * vec_size) {
    AccessT vec = *reinterpret_cast<const AccessT*>(p_SP + idx);
#pragma unroll
    for (int j = 0; j < vec_size; ++j) {
      int global_c = global_c_base + idx + j;
      T val = global_c < kv_seqlen ? vec.val[j] : Traits::fromFloat(-INFINITY);
      partial_max = max(partial_max, Traits::toFloat(val));
    }
  }

  float block_max = MaxBlockReduce(max_temp_storage).Reduce(partial_max, cub::Max());
  if (tid == 0) {
    smem_new_m = block_max;
  }
  __syncthreads();
  new_m = smem_new_m;

  float partial_sum = 0.0f;
  for (int idx = tid * vec_size; idx < bc; idx += thread_per_block * vec_size) {
    AccessT vec = *reinterpret_cast<const AccessT*>(p_SP + idx);
#pragma unroll
    for (int j = 0; j < vec_size; ++j) {
      int global_c = global_c_base + idx + j;
      float p =
          global_c < kv_seqlen ? expf(Traits::toFloat(vec.val[j]) - Traits::toFloat(new_m)) : 0.0f;
      partial_sum += p;
      vec.val[j] = Traits::fromFloat(p);
    }
    *reinterpret_cast<AccessT*>(p_SP + idx) = vec;
  }

  float block_sum = SumBlockReduce(sum_temp_storage).Sum(partial_sum);
  if (tid == 0) {
    smem_l = block_sum;
  }
  __syncthreads();
  l = l * expf(Traits::toFloat(old_m) - Traits::toFloat(new_m)) + smem_l;
}

// P[1, bc] @ V[bc, head_dim] -> O[1, head_dim]
template <typename T, int head_dim>
__device__ __forceinline__ void computeOTileStage(const T* p_P,
                                                  const T* p_V,  // [kv_stage_rows, head_dim]
                                                  float& acc,
                                                  int kv_stage_rows) {
  int tid = threadIdx.x;
  if (tid < head_dim) {
    for (int i = 0; i < kv_stage_rows; ++i) {
      acc += static_cast<float>(p_P[i]) * static_cast<float>(p_V[i * head_dim + tid]);
    }
  }
}

template <typename T, int bc, int head_dim>
__device__ __forceinline__ void updateOTile(const T* p_V_gmem,
                                            const T* p_P_smem,  // [bc]
                                            T* p_V_smem,        // [2, kv_stage_rows, head_dim]
                                            T* p_O_smem,
                                            T old_m,
                                            T new_m,
                                            const int* block_table,
                                            int seq_base,
                                            int kv_seqlen,
                                            int seqlen_stride,
                                            int paged_block_size,
                                            int kv_stage_rows) {
  int tid = threadIdx.x;
  float acc = 0.0f;

  loadKVPagedTileAsync<T, head_dim>(p_V_gmem, p_V_smem, block_table, seq_base, kv_seqlen,
                                    seqlen_stride, paged_block_size);

  CP_ASYNC_WAIT_ALL();
  __syncthreads();

  int seqlen = min(kv_seqlen - seq_base, bc);
  int stages = (seqlen + kv_stage_rows - 1) / kv_stage_rows;

  for (int i = 0; i < stages; i++) {
    int smem_offset_V = (i % 2) * kv_stage_rows * head_dim;
    int next_smem_offset_V = ((i + 1) % 2) * kv_stage_rows * head_dim;
    int p_offset = i * kv_stage_rows;

    if (i + 1 < stages) {
      loadKVPagedTileAsync<T, head_dim>(p_V_gmem, p_V_smem + next_smem_offset_V, block_table,
                                        seq_base + (i + 1) * kv_stage_rows, kv_seqlen,
                                        seqlen_stride, paged_block_size);
    }

    int valid_rows = min(kv_stage_rows, seqlen - i * kv_stage_rows);
    computeOTileStage<T, head_dim>(p_P_smem + p_offset, p_V_smem + smem_offset_V, acc, valid_rows);

    CP_ASYNC_WAIT_ALL();
    __syncthreads();
  }

  if (tid < head_dim) {
    using Traits = NumericTraits<T>;
    float old_O = p_O_smem[tid];
    float m_diff = Traits::toFloat(old_m) - Traits::toFloat(new_m);
    float new_O = old_O * expf(m_diff) + acc;
    p_O_smem[tid] = new_O;
  }
}

template <typename T, int head_dim>
__device__ __forceinline__ void initOTile(T* p_O_smem) {
  int tid = threadIdx.x;
  if (tid < head_dim) {
    p_O_smem[tid] = 0.0f;
  }
}

template <typename T, int head_dim>
__device__ __forceinline__ void storeOTile(T* p_O_smem, T* p_O_gmem, float l) {
  int tid = threadIdx.x;
  if (tid < head_dim) {
    using Traits = NumericTraits<T>;
    float o_val = p_O_smem[tid];
    p_O_gmem[tid] = Traits::fromFloat(o_val / l);
  }
}

// q: [batch, 1, num_heads, head_dim]
template <typename T, int bc, int head_dim, int thread_per_block>
__global__ void GQAVarlenDecodeKernelImpl(const T* __restrict__ q,
                                          const T* __restrict__ k,
                                          const T* __restrict__ v,
                                          T* __restrict__ output,
                                          const int* __restrict__ cu_seqlens_kv,
                                          const int* __restrict__ block_tables,
                                          const int num_heads,
                                          const int kv_heads,
                                          const int block_table_len,
                                          const int paged_block_size) {
  static_assert(head_dim % 64 == 0, "head_dim must be multiple of 64");

  extern __shared__ char smem[];

  int batch_id = blockIdx.x;
  int head_id = blockIdx.y;
  int kv_head_id = head_id / (num_heads / kv_heads);

  int kv_seqlen_stride = kv_heads * head_dim;
  int kv_seqlen = cu_seqlens_kv[batch_id + 1] - cu_seqlens_kv[batch_id];

  const int* block_table = block_tables + batch_id * block_table_len;

  int q_seqlen_stride = num_heads * head_dim;

  int q_gmem_offset = batch_id * q_seqlen_stride + head_id * head_dim;
  int k_gmem_offset = kv_head_id * head_dim;
  int v_gmem_offset = k_gmem_offset;  // seqlen stride should be handled in loadKVPagedTile
  int out_gmem_offset = q_gmem_offset;

  int kv_stage_rows = blockDim.x * 16 / (head_dim * sizeof(T));

  T* p_q_smem = reinterpret_cast<T*>(smem);               // [1, head_dim]
  T* p_k_smem = p_q_smem + head_dim;                      // [2, kv_stage_rows, head_dim]
  T* p_v_smem = p_k_smem + 2 * kv_stage_rows * head_dim;  // [2, kv_stage_rows, head_dim]
  T* p_s_smem = p_v_smem + 2 * kv_stage_rows * head_dim;  // [bc]
  T* p_o_smem = p_s_smem + bc;                            // [head_dim]

  initOTile<T, head_dim>(p_o_smem);
  loadQHead<T, head_dim>(q + q_gmem_offset, p_q_smem);
  __syncthreads();

  T m0 = -INFINITY, m1 = -INFINITY;
  float l = 0.0f;

  for (int c = 0; c < kv_seqlen; c += bc) {
    computeSTile<T, bc, head_dim>(p_q_smem, k + k_gmem_offset, p_k_smem, p_s_smem, block_table, c,
                                  kv_seqlen, kv_seqlen_stride, paged_block_size, kv_stage_rows);
    __syncthreads();

    T& old_m = ((c / bc) & 1) == 0 ? m0 : m1;
    T& new_m = ((c / bc) & 1) == 0 ? m1 : m0;
    computePTile<T, bc, head_dim, thread_per_block>(p_s_smem, old_m, new_m, l, kv_seqlen, c);
    __syncthreads();

    updateOTile<T, bc, head_dim>(v + v_gmem_offset, p_s_smem, p_v_smem, p_o_smem, old_m, new_m,
                                 block_table, c, kv_seqlen, kv_seqlen_stride, paged_block_size,
                                 kv_stage_rows);
    __syncthreads();
  }

  storeOTile<T, head_dim>(p_o_smem, output + out_gmem_offset, l);
}

// q: [batch, 1, num_heads, head_dim]
// k, v: [num_blocks, block_size, num_heads_kv, head_dim] where num_heads_kv divides num_heads
template <typename T, typename Context>
void GQAVarlenDecodeKernel(const Context& ctx,
                           const tensor::Tensor& q,
                           const tensor::Tensor& k,
                           const tensor::Tensor& v,
                           const tensor::Tensor& cu_seqlens_kv,
                           const tensor::Tensor& block_tables,
                           const int paged_block_size,
                           tensor::Tensor& output) {
  CHECK(ctx.getDeviceType() == common::DeviceType::kDeviceCUDA)
      << "GQAVarlenDecodeKernel only supports CUDA device type.";
  const auto& cuda_ctx = static_cast<const common::CUDADeviceContext&>(ctx);

  const auto& q_shape = q.shape();
  const int num_heads = q_shape[1];
  const int head_dim = q_shape[2];

  const auto& k_shape = k.shape();
  const int kv_heads = k_shape[2];

  const int batch = block_tables.shape()[0];
  const int block_table_len = block_tables.shape()[1];

  const T* p_q = q.data<T>();
  const T* p_k = k.data<T>();
  const T* p_v = v.data<T>();
  const int* p_cu_seqlens_kv = cu_seqlens_kv.data<int>();
  const int* p_block_tables = block_tables.data<int>();
  T* p_output = output.data<T>();

  constexpr int bc = 1024;
  constexpr int block_dim = 128;
  dim3 grid_dim(batch, num_heads);

  int kv_stage_rows = block_dim * 16 / (head_dim * sizeof(T));
  size_t smem_size = (head_dim + 4 * kv_stage_rows * head_dim + bc + head_dim) * sizeof(T);

  auto dispatcher = [&]() {
    DLOG(INFO) << "GQA decode kernel smem size: " << smem_size;
    switch (head_dim) {
      case 64:
        cudaFuncSetAttribute(GQAVarlenDecodeKernelImpl<T, bc, 64, block_dim>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
        GQAVarlenDecodeKernelImpl<T, bc, 64, block_dim>
            <<<grid_dim, block_dim, smem_size, cuda_ctx.getStream()>>>(
                p_q, p_k, p_v, p_output, p_cu_seqlens_kv, p_block_tables, num_heads, kv_heads,
                block_table_len, paged_block_size);
        break;
      case 128:
        cudaFuncSetAttribute(GQAVarlenDecodeKernelImpl<T, bc, 128, block_dim>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
        GQAVarlenDecodeKernelImpl<T, bc, 128, block_dim>
            <<<grid_dim, block_dim, smem_size, cuda_ctx.getStream()>>>(
                p_q, p_k, p_v, p_output, p_cu_seqlens_kv, p_block_tables, num_heads, kv_heads,
                block_table_len, paged_block_size);
        break;
      default:
        throw std::runtime_error("Unsupported head_dim in GQAVarlenDecodeKernel dispatcher");
    }
  };

  dispatcher();
}

REGISTER_KERNEL(GQAVarlenDecode, CUDA, GQAVarlenDecodeKernel, Float16, BFloat16);

}  // namespace ginfer::core::op::kernel