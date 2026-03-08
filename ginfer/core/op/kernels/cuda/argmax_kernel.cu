#include <glog/logging.h>
#include "ginfer/core/op/kernels/argmax_kernel.h"
#include "ginfer/core/op/kernels/cuda/vectorize.cuh"
#include "ginfer/core/op/kernels/kernel_registry.h"

namespace ginfer::core::op::kernel {

__forceinline__ __device__ void warp_reduce_argmax(float& val, int64_t& idx) {
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    float other_val = __shfl_down_sync(0xFFFFFFFF, val, offset);
    size_t other_idx = __shfl_down_sync(0xFFFFFFFF, idx, offset);
    if (other_val > val) {
      val = other_val;
      idx = other_idx;
    }
  }
}

__forceinline__ __device__ void block_reduce_argmax(float& val,
                                                    int64_t& idx,
                                                    float* shared_vals,   // 32
                                                    int64_t* shared_idxs  // 32
) {
  int tid = threadIdx.x;
  int lane = tid % 32;
  int warp_id = tid / 32;

  warp_reduce_argmax(val, idx);

  if (lane == 0) {
    shared_vals[warp_id] = val;
    shared_idxs[warp_id] = idx;
  }
  __syncthreads();

  if (tid < blockDim.x / warpSize) {
    val = shared_vals[lane];
    idx = shared_idxs[lane];
  } else {
    val = -INFINITY;
    idx = 0;
  }

  if (warp_id == 0) {
    warp_reduce_argmax(val, idx);
  }

  __syncthreads();
}

template <typename T, int vec_size = DefaultVecSize<T>::value>
__global__ void argmaxKernelImpl(const T* input, int64_t* output_idx, size_t size) {
  __shared__ int64_t max_idx[32];
  __shared__ float max_val[32];

  uint32_t tid = threadIdx.x;

  float thread_max_val = -INFINITY;
  int64_t thread_max_idx = 0;

  using AccessT = AlignedVector<T, vec_size>;
  const AccessT* input_vec = reinterpret_cast<const AccessT*>(input);

  size_t total_vec_size = size / vec_size;
  for (size_t i = tid; i < total_vec_size; i += blockDim.x) {
    const AccessT data = input_vec[i];
#pragma unroll
    for (int j = 0; j < vec_size; j++) {
      float val = static_cast<float>(data.val[j]);
      int64_t idx = i * vec_size + j;
      if (val > thread_max_val) {
        thread_max_val = val;
        thread_max_idx = idx;
      }
    }
  }

  block_reduce_argmax(thread_max_val, thread_max_idx, max_val, max_idx);

  if (threadIdx.x == 0) {
    output_idx[0] = thread_max_idx;
  }
}

template <typename T, typename Context>
void argmaxKernel(const Context& ctx, const tensor::Tensor& input, tensor::Tensor& output_idx) {
  CHECK(ctx.getDeviceType() == common::DeviceType::kDeviceCUDA)
      << "addKernel only supports CUDA device type.";
  auto cuda_ctx = static_cast<const common::CUDADeviceContext&>(ctx);

  size_t size = input.size();

  // Validate input length is multiple of vectorization size on host side.
  // `vec_size` is a compile-time constant derived from T, so use it here.
  static constexpr int vec_size = DefaultVecSize<T>::value;
  CHECK(size % vec_size == 0) << "Input size must be multiple of vector size";

  const T* input_data = input.data<T>();
  int64_t* output_idx_data = output_idx.data<int64_t>();

  const int block_dim = 512;
  const int grid_dim = 1;

  argmaxKernelImpl<T, vec_size>
      <<<grid_dim, block_dim, 0, cuda_ctx.getStream()>>>(input_data, output_idx_data, size);
}

REGISTER_KERNEL(argmax,
                kDeviceCUDA,
                argmaxKernel,
                tensor::DataType::kDataTypeFloat16,
                tensor::DataType::kDataTypeFloat32,
                tensor::DataType::kDataTypeBFloat16);

}  // namespace ginfer::core::op::kernel