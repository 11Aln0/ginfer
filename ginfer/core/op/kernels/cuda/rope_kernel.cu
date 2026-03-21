#include "ginfer/core/op/kernels/kernel_registry.h"
#include "ginfer/core/op/kernels/rope_kernel.h"

namespace ginfer::core::op::kernel {

struct DefaultRopePolicy {
  float rope_theta;
  int half_head_dim;

  DefaultRopePolicy(float rope_theta, int half_head_dim)
      : rope_theta(rope_theta), half_head_dim(half_head_dim) {}

  __device__ __forceinline__ float computeInvFreq(int j) {
    return 1.0f / powf(rope_theta,
                       (float)j / (float)half_head_dim);  // 2 * j / head_dim -> j / rope_half_dim
  }
};

struct Llama3RopePolicy {
  float rope_theta;
  int half_head_dim;
  float factor;
  float low_freq_factor;
  float high_freq_factor;
  int old_ctx_len;

  Llama3RopePolicy(float rope_theta,
                   int half_head_dim,
                   float factor,
                   float low_freq_factor,
                   float high_freq_factor,
                   int old_ctx_len)
      : rope_theta(rope_theta),
        half_head_dim(half_head_dim),
        factor(factor),
        low_freq_factor(low_freq_factor),
        high_freq_factor(high_freq_factor),
        old_ctx_len(old_ctx_len) {}

  __device__ __forceinline__ float computeInvFreq(int j) {
    float low_freq_wavelen = (float)old_ctx_len / low_freq_factor;
    float high_freq_wavelen = (float)old_ctx_len / high_freq_factor;

    float inv_freq = 1.0f / powf(rope_theta, (float)j / (float)half_head_dim);
    float wavelen = 2 * M_PI * inv_freq;

    if (wavelen > low_freq_wavelen) {
      inv_freq = inv_freq / factor;
    } else if (wavelen < high_freq_wavelen) {
    } else {
      float smooth =
          (old_ctx_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor);
      inv_freq = (1 - smooth) * inv_freq / factor + smooth * inv_freq;
    }

    return inv_freq;
  }
};

template <typename T, typename Policy>
__global__ void rotaryEmbeddingImpl(
    Policy policy, T* sin_cache, T* cos_cache, int start_pos, int end_pos, int half_head_dim) {
  int j = threadIdx.x;
  float inv_freq = policy.computeInvFreq(j);

  for (int pos_id = start_pos + blockIdx.x; pos_id <= end_pos; pos_id += gridDim.x) {
    float theta = pos_id * inv_freq;
    // write from the start of the cache
    sin_cache[(pos_id - start_pos) * half_head_dim + j] = static_cast<T>(sinf(theta));
    cos_cache[(pos_id - start_pos) * half_head_dim + j] = static_cast<T>(cosf(theta));
  }
}

template <typename T>
__global__ void ROPEImpl(T* output,
                         const T* input,
                         const int* positions,
                         const float* sin_cache,
                         const float* cos_cache,
                         int total_num_heads,  // seq_len * nhead
                         int num_heads,
                         int head_dim) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  int rope_half_dim = head_dim / 2;

  int j = tid % rope_half_dim;
  int iter_stride = blockDim.x * gridDim.x / rope_half_dim;

  for (int h = tid / rope_half_dim; h < total_num_heads; h += iter_stride) {
    int i = positions[h / num_heads];
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

template <typename T, typename Context>
void RotaryEmbeddingKernel(const Context& ctx,
                           tensor::Tensor& sin_cache,
                           tensor::Tensor& cos_cache,
                           int start_pos,
                           int end_pos,
                           float rope_theta) {
  CHECK(ctx.getDeviceType() == common::DeviceType::kDeviceCUDA)
      << "RotaryEmbeddingKernel only supports CUDA device type.";
  auto cuda_ctx = static_cast<const common::CUDADeviceContext&>(ctx);

  const auto& shape = sin_cache.shape();

  T* sin_cache_data = sin_cache.data<T>();
  T* cos_cache_data = cos_cache.data<T>();

  int half_head_dim = shape[shape.ndim() - 1];
  int block_size = half_head_dim;
  int grid_size = std::min((end_pos - start_pos + 1), 512);

  DefaultRopePolicy policy(rope_theta, half_head_dim);
  rotaryEmbeddingImpl<T><<<grid_size, block_size, 0, cuda_ctx.getStream()>>>(
      policy, sin_cache_data, cos_cache_data, start_pos, end_pos, half_head_dim);
}

// input [seq_len, nhead, head_dim]
// Computes Llama3-scaled sin/cos into caches, then applies RoPE to input -> output
template <typename T, typename Context>
void Llama3RotaryEmbeddingKernel(const Context& ctx,
                                 tensor::Tensor& sin_cache,
                                 tensor::Tensor& cos_cache,
                                 int start_pos,
                                 int end_pos,
                                 float rope_theta,
                                 float factor,
                                 float high_freq_factor,
                                 float low_freq_factor,
                                 int old_ctx_len) {
  CHECK(ctx.getDeviceType() == common::DeviceType::kDeviceCUDA)
      << "RotaryEmbeddingKernel only supports CUDA device type.";
  auto cuda_ctx = static_cast<const common::CUDADeviceContext&>(ctx);

  const auto& shape = sin_cache.shape();

  T* sin_cache_data = sin_cache.data<T>();
  T* cos_cache_data = cos_cache.data<T>();

  int half_head_dim = shape[shape.ndim() - 1];
  int block_size = half_head_dim;
  int grid_size = std::min((end_pos - start_pos + 1), 512);

  Llama3RopePolicy policy(rope_theta, half_head_dim, factor, low_freq_factor, high_freq_factor,
                          old_ctx_len);
  rotaryEmbeddingImpl<T><<<grid_size, block_size, 0, cuda_ctx.getStream()>>>(
      policy, sin_cache_data, cos_cache_data, start_pos, end_pos, half_head_dim);
}

// input: [seq_len, nhead, head_dim]
// positions: [seq_len]
// sin_cache/cos_cache: [max_position_embeddings, head_dim / 2]
template <typename T, typename Context>
void ROPEKernel(const Context& ctx,
                const tensor::Tensor& input,
                const tensor::Tensor& positions,
                const tensor::Tensor& sin_cache,
                const tensor::Tensor& cos_cache,
                tensor::Tensor& output) {
  CHECK(ctx.getDeviceType() == common::DeviceType::kDeviceCUDA)
      << "ROPEKernel only supports CUDA device type.";

  auto cuda_ctx = static_cast<const common::CUDADeviceContext&>(ctx);

  const auto& input_shape = input.shape();
  int shape_dim = input_shape.ndim();
  int head_dim = input_shape[shape_dim - 1];
  int num_heads = input_shape[shape_dim - 2];
  int total_num_heads =
      std::accumulate(input_shape.begin(), input_shape.end() - 1, 1, std::multiplies<int>());

  CHECK(positions.shape()[0] == input_shape[0])
      << "Positions tensor must have the same sequence length as input.";

  const T* input_data = input.data<T>();
  const int* positions_data = positions.data<int>();
  const float* sin_cache_data = sin_cache.data<float>();
  const float* cos_cache_data = cos_cache.data<float>();
  T* output_data = output.data<T>();

  int block_size = 256;
  int grid_size = std::min((total_num_heads * (head_dim / 2) + block_size - 1) / block_size, 512);

  ROPEImpl<T><<<grid_size, block_size, 0, cuda_ctx.getStream()>>>(
      output_data, input_data, positions_data, sin_cache_data, cos_cache_data, total_num_heads,
      num_heads, head_dim);
}

REGISTER_KERNEL(rotary_embedding,
                kDeviceCUDA,
                RotaryEmbeddingKernel,
                tensor::DataType::kDataTypeFloat32);

REGISTER_KERNEL(llama3_rotary_embedding,
                kDeviceCUDA,
                Llama3RotaryEmbeddingKernel,
                tensor::DataType::kDataTypeFloat32);

REGISTER_KERNEL(ROPE,
                kDeviceCUDA,
                ROPEKernel,
                tensor::DataType::kDataTypeFloat32,
                tensor::DataType::kDataTypeFloat16,
                tensor::DataType::kDataTypeBFloat16);

}  // namespace ginfer::core::op::kernel