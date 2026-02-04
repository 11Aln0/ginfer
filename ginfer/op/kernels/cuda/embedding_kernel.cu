#include "ginfer/op/kernels/embedding_kernel.h"
#include "ginfer/op/kernels/kernel_registry.h"
#include "ginfer/op/kernels/cuda/vectorize.cuh"

namespace ginfer::op::kernel {

template <typename T, int VecSize = DefaultVecSize<T>::value>
__global__ void embeddingKernelImpl(const int64_t* input, const T* weight, T* output,
                                    size_t num_indices, int64_t embedding_dim) {

  using AccessT = AlignedVector<T, VecSize>;

  int tid = threadIdx.x;
  int token_idx =  blockIdx.x;
  if(token_idx >= num_indices) return;

  int64_t token_id = input[token_idx];

  int bound = embedding_dim / VecSize;
  AccessT* out_vec = reinterpret_cast<AccessT*>(output + token_idx * embedding_dim);
  const AccessT* weight_vec = reinterpret_cast<const AccessT*>(weight + token_id * embedding_dim);

  for(int i = tid; i < bound; i += blockDim.x) {
    out_vec[i] = weight_vec[i];
  }
}

template<typename T, typename Context>
void embeddingKernel(const Context& ctx, const tensor::Tensor& input, 
               const tensor::Tensor& weight, tensor::Tensor& output) {
  CHECK(ctx.getDeviceType() == common::DeviceType::kDeviceCUDA)
      << "embeddingKernel only supports CUDA device type.";
  
  auto cuda_ctx = static_cast<const common::CUDADeviceContext&>(ctx);
  const auto& input_shape = input.shape();
  const auto& weight_shape = weight.shape();
  size_t num_indices = std::accumulate(input_shape.begin(), input_shape.end(), 1, std::multiplies<size_t>());

  const int64_t* input_data = input.data<int64_t>();
  const T* weight_data = weight.data<T>();
  T* output_data = output.data<T>();

  size_t embedding_dim = weight_shape[weight_shape.ndim() - 1];
  int block_size = 128;

  embeddingKernelImpl<T><<<num_indices, block_size, 0, cuda_ctx.getStream()>>>(
      input_data, weight_data, output_data,
      num_indices, embedding_dim);

}

REGISTER_KERNEL(embedding,
                kDeviceCUDA,
                embeddingKernel,
                tensor::DataType::kDataTypeFloat16,
                tensor::DataType::kDataTypeFloat32);


} // namespace ginfer::op::kernel