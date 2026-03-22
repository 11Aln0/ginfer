#include <cub/block/block_reduce.cuh>
#include "ginfer/common/check.h"
#include "ginfer/core/op/kernels/cuda/vectorize.cuh"
#include "ginfer/core/op/kernels/kernel_registry.h"
#include "ginfer/core/op/kernels/kernels.h"
#include "ginfer/core/tensor/tensor.h"

namespace ginfer::core::op::kernel {

template <typename T,
          int vec_size = DefaultVecSize<T>::value,
          int thread_per_block = 128,
          bool has_bias = false>
__global__ void gemvKernelImpl(const T* __restrict__ vec,
                               const T* __restrict__ mat,
                               const T* __restrict__ bias,
                               T* __restrict__ output,
                               const int K,
                               const int N) {
  using AccessT = AlignedVector<T, vec_size>;

  const int pack_K = K / vec_size;

  const AccessT* p_vec = reinterpret_cast<const AccessT*>(vec);
  const AccessT* p_mat = reinterpret_cast<const AccessT*>(mat);

  using BlockReduce = cub::BlockReduce<float, thread_per_block>;
  __shared__ typename BlockReduce::TempStorage temp;

  // r: row of mat; c: col of mat

#pragma unroll
  for (int c = blockIdx.x; c < N; c += gridDim.x) {
    float thread_sum = 0.0f;
    int mat_base = c * K;
    p_mat = reinterpret_cast<const AccessT*>(mat + mat_base);

    for (int r = threadIdx.x; r < pack_K; r += blockDim.x) {
      AccessT vec_val = p_vec[r];
      AccessT mat_val = p_mat[r];
      thread_sum += DotProduct<T, vec_size>::run(vec_val, mat_val);
    }

    for (int r = pack_K * vec_size + threadIdx.x; r < K; r += blockDim.x) {
      thread_sum += static_cast<float>(vec[r]) * static_cast<float>(mat[mat_base + r]);
    }

    __syncthreads();

    float block_sum = BlockReduce(temp).Sum(thread_sum);

    if (threadIdx.x == 0) {
      if constexpr (has_bias) {
        block_sum += static_cast<float>(bias[c % N]);
      }
      output[c] = static_cast<T>(block_sum);
    }
  }
}

template <typename T, typename Context>
void gemvKernel(const Context& ctx,
                const tensor::Tensor& vec,
                const tensor::Tensor& mat,
                std::optional<std::reference_wrapper<const tensor::Tensor>> bias,
                tensor::Tensor& output) {
  CHECK(ctx.getDeviceType() == common::DeviceType::kDeviceCUDA)
      << "gemvKernel only supports CUDA device type.";

  auto cuda_ctx = static_cast<const common::CUDADeviceContext&>(ctx);

  const auto& shape = mat.shape();
  const auto& strides = mat.strides();
  CHECK(shape.ndim() == 2) << "gemvKernel only supports 2D tensors.";
  CHECK(strides[1] == shape[0]) << "Matrix must be in column-major order.";
  const auto K = shape[0];
  const auto N = shape[1];

  const T* vec_data = vec.data<T>();
  const T* mat_data = mat.data<T>();
  T* output_data = output.data<T>();

  const int block_dim = 128;
  const int grid_dim = std::min(N, 512L);

  if (bias.has_value()) {
    const T* bias_data = bias->get().data<T>();
    gemvKernelImpl<T, DefaultVecSize<T>::value, block_dim, true>
        <<<grid_dim, block_dim, 0, cuda_ctx.getStream()>>>(vec_data, mat_data, bias_data,
                                                           output_data, K, N);
  } else {
    gemvKernelImpl<T, DefaultVecSize<T>::value, block_dim, false>
        <<<grid_dim, block_dim, 0, cuda_ctx.getStream()>>>(vec_data, mat_data, nullptr, output_data,
                                                           K, N);
  }
}

REGISTER_KERNEL(gemv,
                kDeviceCUDA,
                gemvKernel,
                tensor::DataType::kDataTypeFloat32,
                tensor::DataType::kDataTypeFloat16,
                tensor::DataType::kDataTypeBFloat16);

}  // namespace ginfer::core::op::kernel