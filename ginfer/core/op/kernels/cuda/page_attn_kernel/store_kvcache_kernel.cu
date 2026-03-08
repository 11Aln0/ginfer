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
}

template <typename T, typename Context>
void storeKVCacheKernel(const Context& ctx,
                        const tensor::Tensor& k,
                        const tensor::Tensor& v,
                        tensor::Tensor& k_cache,
                        tensor::Tensor& v_cache,
                        const tensor::Tensor& slot_mapping) {
  CHECK(ctx.getDeviceType() == common::DeviceType::kDeviceCUDA)
      << "embeddingKernel only supports CUDA device type.";

  auto cuda_ctx = static_cast<const common::CUDADeviceContext&>(ctx);
  const auto& k_shape = k.shape();
  const auto& k_stride = k.strides();

  int N = k_shape[0];
  int stride = k_stride[0];

  int block_size = 128;
  int grid_size = N;

  storeKVCacheKernelImpl<T><<<grid_size, block_size, 0, cuda_ctx.getStream()>>>(
      k.data<T>(), v.data<T>(), k_cache.data<T>(), v_cache.data<T>(), slot_mapping.data<int>(),
      stride);
}

REGISTER_KERNEL(storeKVCache,
                kDeviceCUDA,
                storeKVCacheKernel,
                tensor::DataType::kDataTypeBFloat16,
                tensor::DataType::kDataTypeFloat16);

}  // namespace ginfer::core::op::kernel