#include <cuda_runtime.h>
#include <glog/logging.h>
#include "ginfer/core/op/kernels/cuda/intrinsic.cuh"
#include "ginfer/core/op/kernels/cuda/vectorize.cuh"
#include "ginfer/core/op/kernels/kernel_registry.h"
#include "ginfer/core/op/kernels/page_attn_kernel.h"

namespace ginfer::core::op::kernel {

template <typename T, int vec_size = DefaultVecSize<T>::value>
__global__ void storeKVCacheKernelImpl(const T* __restrict__ p_k,
                                       const T* __restrict__ p_v,
                                       T* __restrict__ p_k_cache,
                                       T* __restrict__ p_v_cache,
                                       const int* __restrict__ slot_mapping,
                                       const int stride) {
  using AccessT = AlignedVector<T, vec_size>;
  int tid = threadIdx.x;
  int slot = slot_mapping[blockIdx.x];

  const AccessT* p_k_vec = reinterpret_cast<const AccessT*>(p_k + blockIdx.x * stride);
  const AccessT* p_v_vec = reinterpret_cast<const AccessT*>(p_v + blockIdx.x * stride);
  AccessT* p_k_cache_vec = reinterpret_cast<AccessT*>(p_k_cache + slot * stride);
  AccessT* p_v_cache_vec = reinterpret_cast<AccessT*>(p_v_cache + slot * stride);

  int bound = stride / vec_size;
  for (int i = tid; i < bound; i += blockDim.x) {
    p_k_cache_vec[i] = p_k_vec[i];
    p_v_cache_vec[i] = p_v_vec[i];
  }

  // if (blockIdx.x == 0 && tid == 0) {
  //   printf("-----------------storeKVCacheKernelImpl-----------------\n");
  //   for (int i = 0; i < gridDim.x; i++) {
  //     printf("p_k[%d] = %f, p_k_cache[%d] = %f\n", i, static_cast<float>(p_k[i * stride]), i,
  //            static_cast<float>(p_k_cache[i * stride]));
  //     printf("p_v[%d] = %f, p_v_cache[%d] = %f\n", i, static_cast<float>(p_v[i * stride]), i,
  //            static_cast<float>(p_v_cache[i * stride]));
  //   }
  // }
}

// k/v_cache: [num_blocks, block_size, num_kv_heads, head_dim]
// k/v: [seqlen, num_kv_heads, head_dim]
template <typename T, typename Context>
void storeKVCacheKernel(const Context& ctx,
                        const tensor::Tensor& k,
                        const tensor::Tensor& v,
                        tensor::Tensor& k_cache,
                        tensor::Tensor& v_cache,
                        const tensor::Tensor& slot_mapping) {
  CHECK(ctx.getDeviceType() == common::DeviceType::kDeviceCUDA)
      << "embeddingKernel only supports CUDA device type.";

  const auto& cuda_ctx = static_cast<const common::CUDADeviceContext&>(ctx);
  const auto& k_shape = k.shape();
  const auto& k_stride = k.strides();

  int seq_len = k_shape[0];
  int stride = k_stride[0];

  int block_size = 128;
  int grid_size = seq_len;

  storeKVCacheKernelImpl<T><<<grid_size, block_size, 0, cuda_ctx.getStream()>>>(
      k.data<T>(), v.data<T>(), k_cache.data<T>(), v_cache.data<T>(), slot_mapping.data<int>(),
      stride);
}

REGISTER_KERNEL(storeKVCache, CUDA, storeKVCacheKernel, BFloat16, Float16);

}  // namespace ginfer::core::op::kernel