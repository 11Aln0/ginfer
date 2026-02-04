#include "ginfer/op/kernels/kernel_registry.h"
#include "ginfer/op/kernels/rope_kernel.h"

namespace ginfer::op::kernel {

template<typename T>
__global__ void calcSinCosImpl(T* sin_cache,
                               T* cos_cache,
                               int start_pos,
                               int end_pos,
                               int head_dim,
                               float rope_theta) {

  int j = threadIdx.x;
  int rope_half_dim = head_dim / 2;

  for(int pos_id = start_pos + blockIdx.x; pos_id < end_pos; pos_id += gridDim.x) {
    float theta = pos_id / powf(rope_theta, (2.0f * (float)j) / (float)head_dim);
    // write from the start of the cache
    sin_cache[(pos_id - start_pos) * rope_half_dim + j] = static_cast<T>(sinf(theta));
    cos_cache[(pos_id - start_pos) * rope_half_dim + j] = static_cast<T>(cosf(theta));
  }
} 


template<typename T>
__global__ void ROPEImpl(T* output,
                         const T* input,
                         const float* sin_cache,
                         const float* cos_cache,
                         int total_num_heads, // seq_len * nhead
                         int num_heads,
                         int head_dim) {
  
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  int rope_half_dim = head_dim / 2;

  int j = tid % rope_half_dim;
  int iter_stride = blockDim.x * gridDim.x / rope_half_dim;

  for(int h = tid / rope_half_dim; h < total_num_heads; h += iter_stride) {
    int i = h / num_heads;
    float sin_val = sin_cache[i * rope_half_dim + j];
    float cos_val = cos_cache[i * rope_half_dim + j];

    int even_off = h * head_dim + j;
    int odd_off = h * head_dim + rope_half_dim + j;

    float input_even = static_cast<float>(input[even_off]);
    float input_odd = static_cast<float>(input[odd_off]);

    output[even_off] = static_cast<T>(input_even * cos_val - input_odd * sin_val);
    output[odd_off] = static_cast<T>(input_even * sin_val + input_odd * cos_val);
  }
}

template<typename T, typename Context>
void CalcSinCosKernel(const Context& ctx, 
                      tensor::Tensor sin_cache, tensor::Tensor cos_cache,
                      int start_pos, int end_pos,
                      int head_dim, float rope_theta) {

  CHECK(ctx.getDeviceType() == common::DeviceType::kDeviceCUDA)
      << "CalcSinCostKernel only supports CUDA device type.";

  auto cuda_ctx = static_cast<const common::CUDADeviceContext&>(ctx);

  T* sin_cache_data = sin_cache.data<T>();
  T* cos_cache_data = cos_cache.data<T>();

  int rope_half_dim = head_dim / 2;
  int block_size = rope_half_dim;
  int grid_size = std::min((end_pos - start_pos), 512);

  calcSinCosImpl<T><<<grid_size, block_size, 0, cuda_ctx.getStream()>>>(
      sin_cache_data,
      cos_cache_data,
      start_pos,
      end_pos,
      head_dim,
      rope_theta
  );
}

// input [seq_len, nhead, head_dim]
template<typename T, typename Context>
void ROPEKernel(const Context& ctx, 
                const tensor::Tensor input, tensor::Tensor output,
                const tensor::Tensor sin_cache, tensor::Tensor cos_cache) {

  CHECK(ctx.getDeviceType() == common::DeviceType::kDeviceCUDA)
      << "ROPEKernel only supports CUDA device type.";
  
  auto cuda_ctx = static_cast<const common::CUDADeviceContext&>(ctx);

  const auto& input_shape = input.shape();
  int shape_dim = input_shape.ndim();
  int head_dim = input_shape[shape_dim - 1];
  int num_heads = input_shape[shape_dim - 2];
  int total_num_heads = std::accumulate(input_shape.begin(), input_shape.end() - 1, 1, std::multiplies<int>());

  const T* input_data = input.data<T>();
  T* output_data = output.data<T>();
  const float* sin_cache_data = sin_cache.data<float>();
  const float* cos_cache_data = cos_cache.data<float>();

  int block_size = 256;
  int grid_size = std::min((total_num_heads + block_size - 1) / block_size, 512);

  ROPEImpl<T><<<grid_size, block_size, 0, cuda_ctx.getStream()>>>(
      output_data,
      input_data,
      sin_cache_data,
      cos_cache_data,
      total_num_heads,
      num_heads,
      head_dim
  );
}

REGISTER_KERNEL(calcSinCos,
                kDeviceCUDA,
                CalcSinCosKernel,
                tensor::DataType::kDataTypeFloat32);

REGISTER_KERNEL(ROPE,
                kDeviceCUDA,
                ROPEKernel,
                tensor::DataType::kDataTypeFloat32,
                tensor::DataType::kDataTypeFloat16);

} // namespace ginfer::op::kernel