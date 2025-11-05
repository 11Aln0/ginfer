#include "ginfer/op/kernels/rmsnorm_kernel.h"
#include "ginfer/op/kernels/kernel_registry.h"
#include "ginfer/op/kernels/cuda/vectorize.cuh"
#include <cub/block/block_reduce.cuh>

namespace ginfer::op::kernel {

template <typename T, int vec_size = DefaultVecSize<T>::value, int BlockDim = 256>
__global__ void rmsNormKernelImpl(const T* __restrict__ in, 
                                  const T* __restrict__ gamma, 
                                  T* __restrict__ out, 
                                  const int hidden_dim, 
                                  const float eps) {
  using AccessT = AlignedVector<T, vec_size>;

  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  const int pack_dim = hidden_dim / vec_size;
  const int offset = row * hidden_dim;

  const AccessT* in_vec_ptr = reinterpret_cast<const AccessT*>(in + offset);
  const AccessT* gamma_vec_ptr = reinterpret_cast<const AccessT*>(gamma);
  AccessT* out_vec_ptr = reinterpret_cast<AccessT*>(out + offset);

  float sum = 0.0f;
  
  for(int i = tid; i < pack_dim; i += blockDim.x) {
    AccessT in_vec = in_vec_ptr[i];
    #pragma unroll
    for(int j = 0; j < vec_size; ++j) {
      float v = static_cast<float>(in_vec.val[j]);
      sum += v * v;
    }
  }

  for(int i = pack_dim*vec_size + tid; i < hidden_dim; i += blockDim.x) {
    float v = static_cast<float>(in[offset + i]);
    sum += v * v;
  }

  // Block-wide reduction to compute the total sum
  using BlockReduce = cub::BlockReduce<float, BlockDim>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  __shared__ float shared_sum;
  sum = BlockReduce(temp_storage).Sum(sum);
  if(tid == 0) {
    shared_sum = sum;
  }
  __syncthreads();
  sum = shared_sum;

  float rms = rsqrtf(sum / static_cast<float>(hidden_dim) + eps);
  for(int i = tid; i < pack_dim; i += blockDim.x) {
    AccessT in_vec = in_vec_ptr[i];
    AccessT gamma_vec = gamma_vec_ptr[i];
    AccessT out_vec;
    #pragma unroll
    for(int j = 0; j < vec_size; ++j) {
      float v = static_cast<float>(in_vec.val[j]);
      float g = static_cast<float>(gamma_vec.val[j]);
      out_vec.val[j] = static_cast<T>(v * rms * g);
    }
    out_vec_ptr[i] = out_vec;
  }

  for(int i = pack_dim*vec_size + tid; i < hidden_dim; i += blockDim.x) {
    float v = static_cast<float>(in[offset + i]);
    float g = static_cast<float>(gamma[i]);
    out[offset + i] = static_cast<T>(v * rms * g);
  }
}


template <typename T, typename Context>
void rmsNormKernel(const Context& ctx, 
                   const tensor::Tensor& input, 
                   const tensor::Tensor& gamma, 
                   tensor::Tensor& output, 
                   float epsilon) {
  CHECK(ctx.getDeviceType() == common::DeviceType::kDeviceCUDA)
      << "addKernel only supports CUDA device type.";

  const auto& shape = input.shape();
  int hidden_dim = static_cast<int>(shape[shape.ndim() - 1]);
  int batch_size = static_cast<int>(input.size() / hidden_dim);
  const T* in_data = input.data<T>();
  const T* gamma_data = gamma.data<T>();
  T* out_data = output.data<T>();

  auto cuda_ctx = static_cast<const common::CUDADeviceContext&>(ctx);
  const int block_dim = 256;
  const int grid_dim = batch_size;
  if(cuda_ctx.getStream() == nullptr) {
    rmsNormKernelImpl<T><<<grid_dim, block_dim>>>(in_data, gamma_data, out_data, hidden_dim, epsilon);
  } else {
    rmsNormKernelImpl<T><<<grid_dim, block_dim, 0, cuda_ctx.getStream()>>>(in_data, gamma_data, out_data, hidden_dim, epsilon);
  }
}

REGISTER_KERNEL(rmsNorm,
                kDeviceCUDA,
                rmsNormKernel,
                tensor::DataType::kDataTypeFloat32,
                tensor::DataType::kDataTypeFloat16);

} // namespace ginfer::op::kernel