#include "ginfer/common/check.h"
#include "ginfer/core/op/kernels/cuda/vectorize.cuh"
#include "ginfer/core/op/kernels/kernel_registry.h"

namespace ginfer::core::op::kernel {

template <typename T, int VecSize = DefaultVecSize<T>::value>
__global__ void swigluImpl(T* output, const T* gate, const T* up, int64_t numel) {
  using AccessT = AlignedVector<T, VecSize>;

  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t iter_stride = blockDim.x * gridDim.x;

  AccessT* output_vec = reinterpret_cast<AccessT*>(output);
  const AccessT* p_gate = reinterpret_cast<const AccessT*>(gate);
  const AccessT* p_up = reinterpret_cast<const AccessT*>(up);
  size_t vec_numel = numel / VecSize;

  for (size_t i = tid; i < vec_numel; i += iter_stride) {
    AccessT gate_vec = p_gate[i];
    AccessT up_vec = p_up[i];

#pragma unroll
    for (int v = 0; v < VecSize; v++) {
      float g = gate_vec.val[v];
      float sigmoid_g = 1.0f / (1.0f + exp(static_cast<float>(-g)));
      float swish_g = sigmoid_g * g;
      gate_vec.val[v] = static_cast<T>(swish_g * static_cast<float>(up_vec.val[v]));
    }
    output_vec[i] = gate_vec;
  }
}

template <typename T, typename Context>
void swigluKernel(const Context& ctx,
                  tensor::Tensor& output,
                  const tensor::Tensor& gate,
                  const tensor::Tensor& up) {
  CHECK(ctx.getDeviceType() == common::DeviceType::kDeviceCUDA)
      << "addKernel only supports CUDA device type.";

  auto cuda_ctx = static_cast<const common::CUDADeviceContext&>(ctx);

  int64_t numel = output.size();

  T* output_data = output.data<T>();
  const T* gate_data = gate.data<T>();
  const T* up_data = up.data<T>();

  int block_size = 256;
  int grid_size = std::min((numel / DefaultVecSize<T>::value + block_size - 1) / block_size, 1024L);

  swigluImpl<<<grid_size, block_size, 0, cuda_ctx.getStream()>>>(output_data, gate_data, up_data,
                                                                 numel);
}

REGISTER_KERNEL(swiglu, CUDA, swigluKernel, Float32, Float16, BFloat16);

}  // namespace ginfer::core::op::kernel