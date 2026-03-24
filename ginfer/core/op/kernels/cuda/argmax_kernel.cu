#include <glog/logging.h>
#include "ginfer/core/op/kernels/cuda/vectorize.cuh"
#include "ginfer/core/op/kernels/kernel_registry.h"
#include "ginfer/core/op/kernels/kernels.h"

namespace ginfer::core::op::kernel {

__forceinline__ __device__ void warp_reduce_argmax(float& val, int64_t& idx) {
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    float other_val = __shfl_down_sync(0xFFFFFFFF, val, offset);
    int64_t other_idx = __shfl_down_sync(0xFFFFFFFF, idx, offset);
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

template <typename InT, typename OutT, int vec_size = DefaultVecSize<InT>::value>
__device__ void argmaxInner(const InT* input, OutT* output_idx, size_t inner_dim) {
  __shared__ int64_t max_idx[32];
  __shared__ float max_val[32];

  float thread_max_val = -INFINITY;
  int64_t thread_max_idx = 0;

  using AccessT = AlignedVector<InT, vec_size>;
  const AccessT* input_vec = reinterpret_cast<const AccessT*>(input);

  size_t bound = inner_dim / vec_size;
  for (size_t i = threadIdx.x; i < bound; i += blockDim.x) {
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
    *output_idx = static_cast<OutT>(thread_max_idx);
  }
}

template <typename InT, typename OutT, int vec_size>
__global__ void argmax1DKernelImpl(const InT* input, OutT* output_idx, size_t size) {
  argmaxInner<InT, OutT, vec_size>(input, output_idx, size);
}

template <typename InT, typename OutT, int vec_size>
__global__ void argmax2DKernelImpl(const InT* input,
                                   OutT* output_idx,
                                   size_t outer_dim,
                                   size_t inner_dim) {
  int64_t outer_idx = blockIdx.x;
  const InT* outer_input = input + outer_idx * inner_dim;
  OutT* outer_output_idx = output_idx + outer_idx;
  argmaxInner<InT, OutT, vec_size>(outer_input, outer_output_idx, inner_dim);
}

template <typename InT, typename OutT, typename Context>
void argmaxKernel(const Context& ctx, const tensor::Tensor& input, tensor::Tensor& output_idx) {
  CHECK(ctx.getDeviceType() == common::DeviceType::kDeviceCUDA)
      << "argmaxKernel only supports CUDA device type.";
  auto cuda_ctx = static_cast<const common::CUDADeviceContext&>(ctx);

  constexpr int vec_size = DefaultVecSize<InT>::value;

  const auto& shape = input.shape();
  int ndim = shape.ndim();
  CHECK(ndim == 1 || ndim == 2) << "Only 1D and 2D tensors are supported.";

  const InT* input_data = input.data<InT>();
  OutT* output_idx_data = output_idx.data<OutT>();

  if (ndim == 1) {
    size_t size = input.size();
    CHECK(size % vec_size == 0) << "Input size must be a multiple of vector size for this kernel.";

    const int block_dim = 512;
    const int grid_dim = 1;

    argmax1DKernelImpl<InT, OutT, vec_size>
        <<<grid_dim, block_dim, 0, cuda_ctx.getStream()>>>(input_data, output_idx_data, size);
  } else {
    size_t outer_dim = shape[0];
    size_t inner_dim = shape[1];
    CHECK(inner_dim % vec_size == 0)
        << "Inner dimension must be a multiple of vector size for this kernel.";

    const int block_dim = 128;
    const int grid_dim = static_cast<int>(outer_dim);

    argmax2DKernelImpl<InT, OutT, vec_size><<<grid_dim, block_dim, 0, cuda_ctx.getStream()>>>(
        input_data, output_idx_data, outer_dim, inner_dim);
  }
}

REGISTER_KERNEL_DIFF_IO(argmax,
                        CUDA,
                        argmaxKernel,
                        (Float16, Int32),
                        (BFloat16, Int32),
                        (Float16, Int64),
                        (BFloat16, Int64),
                        (Float32, Int64));
;

}  // namespace ginfer::core::op::kernel