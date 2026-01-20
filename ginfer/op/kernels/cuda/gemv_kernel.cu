#include "ginfer/tensor/tensor.h"
#include <cub/block/block_reduce.cuh>
#include "ginfer/op/kernels/cuda/vectorize.cuh"
#include <cub/block/block_reduce.cuh>
#include "ginfer/common/check.h"
#include "ginfer/op/kernels/kernel_registry.h"
#include "ginfer/op/kernels/gemv_kernel.h"

namespace ginfer::op::kernel {


template <typename T, int vec_size = DefaultVecSize<T>::value, int thread_per_block = 128>
__global__ void gemvKernelImpl(const T* __restrict__ mat,
                               const T* __restrict__ vec,
                               T* __restrict__  output,
                               const int M,
                               const int K) {

  using AccessT = AlignedVector<T, vec_size>;
  
  int row_per_block = (M + gridDim.x - 1) / gridDim.x;
  int start_row = blockIdx.x * row_per_block;
  int end_row = min(start_row + row_per_block, M);
  const int pack_K = K / vec_size;

  const AccessT* p_vec = reinterpret_cast<const AccessT*>(vec);
  const AccessT* p_mat = reinterpret_cast<const AccessT*>(mat);

  using BlockReduce = cub::BlockReduce<float, thread_per_block>;
  __shared__ typename BlockReduce::TempStorage temp;

  for(int r = start_row; r < end_row; r++) {
    float thread_sum = 0.0f;
    int mat_base = r * K;
    p_mat = reinterpret_cast<const AccessT*>(mat + mat_base);

#pragma unroll
    for(int c = threadIdx.x; c < pack_K; c += blockDim.x) {
      AccessT vec_val = p_vec[c];
      AccessT mat_val = p_mat[c];
      thread_sum += DotProduct<T, vec_size>::run(vec_val, mat_val);
    }

    for(int c = pack_K * vec_size + threadIdx.x; c < K; c += blockDim.x) {
      thread_sum += static_cast<float>(vec[c]) * static_cast<float>(mat[mat_base + c]);
    }

    __syncthreads();

    float block_sum = BlockReduce(temp).Sum(thread_sum);

    if(threadIdx.x == 0) {
      output[r] = static_cast<T>(block_sum);
    }

  }
}


template <typename T, typename Context>
void gemvKernel(const Context& ctx,
                const tensor::Tensor& mat,
                const tensor::Tensor& vec,
                tensor::Tensor& output) {
  CHECK(ctx.getDeviceType() == common::DeviceType::kDeviceCUDA)
      << "gemvKernel only supports CUDA device type.";
  CHECK(mat.layout() == tensor::Layout::kLayoutColMajor)
      << "gemvKernel only supports col-major matrix layout.";
  
  auto cuda_ctx = static_cast<const common::CUDADeviceContext&>(ctx);
  
  const auto& shape = mat.shape();
  const auto M = shape[0];
  const auto K = shape[1];
  const T* vec_data = vec.data<T>();
  const T* mat_data = mat.data<T>();
  T* output_data = output.data<T>();

  constexpr int row_per_block = 1;
  const int block_dim = 128;
  const int grid_dim = (M + row_per_block - 1) / row_per_block;

  if(cuda_ctx.getStream() == nullptr) {
    gemvKernelImpl<T><<<grid_dim, block_dim>>>(mat_data, vec_data, output_data, M, K);
  } else {
    gemvKernelImpl<T><<<grid_dim, block_dim, 0, cuda_ctx.getStream()>>>(mat_data, vec_data, output_data, M, K);
  }

}

REGISTER_KERNEL(gemv,
                kDeviceCUDA,
                gemvKernel,
                tensor::DataType::kDataTypeFloat32,
                tensor::DataType::kDataTypeFloat16);  


} // namespace ginfer::op::kernel