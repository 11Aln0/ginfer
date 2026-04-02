#include <glog/logging.h>
#include "ginfer/core/op/kernels/cuda/vectorize.cuh"
#include "ginfer/core/op/kernels/kernel_registry.h"
#include "ginfer/core/op/kernels/kernels.h"

namespace ginfer::core::op::kernel {

template <typename T, int vec_size = DefaultVecSize<T>::value>
__global__ void selectLastTokenKernelImpl(const T* input,
                                          const int* cu_seqlen_q,
                                          T* output,
                                          size_t hidden_dim) {
  using AccessT = AlignedVector<T, vec_size>;

  int batch_id = blockIdx.x;
  int idx = cu_seqlen_q[batch_id + 1] - 1;  // last token index for this batch
  size_t offset = idx * hidden_dim;         // offset to the last token's hidden state

  const AccessT* input_vec = reinterpret_cast<const AccessT*>(input + offset);
  AccessT* output_vec =
      reinterpret_cast<AccessT*>(output + static_cast<size_t>(batch_id) * hidden_dim);

  int bound = hidden_dim / vec_size;
  for (int i = threadIdx.x; i < bound; i += blockDim.x) {
    output_vec[i] = input_vec[i];
  }
}

template <typename T, typename Context>
void selectLastTokenKernel(const Context& ctx,
                           const tensor::Tensor& input,
                           const tensor::Tensor& cu_seqlen_q,
                           tensor::Tensor& output) {
  CHECK(ctx.getDeviceType() == common::DeviceType::kDeviceCUDA)
      << "selectLastTokenKernel only supports CUDA device type.";
  CHECK(input.shape().ndim() == 2) << "input must be 2D [total_tokens, hidden_dim].";
  CHECK(cu_seqlen_q.shape().ndim() == 1) << "cu_seqlen_q must be 1D [batch + 1].";
  CHECK(cu_seqlen_q.dtype() == tensor::DataType::kDataTypeInt32)
      << "cu_seqlen_q dtype must be int32.";

  const auto& cuda_ctx = static_cast<const common::CUDADeviceContext&>(ctx);
  auto batch_size = static_cast<int>(cu_seqlen_q.shape()[0]) - 1;
  auto hidden_dim = input.shape()[1];

  constexpr int vec_size = DefaultVecSize<T>::value;

  CHECK_GE(batch_size, 0) << "Invalid cu_seqlen_q shape.";
  CHECK(output.shape().ndim() == 2) << "output must be 2D [batch, hidden_dim].";
  CHECK(output.shape()[0] == batch_size)
      << "output batch dimension must equal cu_seqlen_q.shape[0] - 1.";
  CHECK(output.shape()[1] == hidden_dim) << "output hidden_dim must match input hidden_dim.";
  CHECK(hidden_dim % vec_size == 0)
      << "hidden_dim must be a multiple of vector size for this kernel.";

  if (batch_size == 0) return;

  const T* input_data = input.data<T>();
  const int* cu_seqlen_q_data = cu_seqlen_q.data<int>();
  T* output_data = output.data<T>();

  const int block_dim = 128;
  const int grid_dim = batch_size;
  selectLastTokenKernelImpl<T><<<grid_dim, block_dim, 0, cuda_ctx.getStream()>>>(
      input_data, cu_seqlen_q_data, output_data, hidden_dim);
}

REGISTER_KERNEL(selectLastToken, CUDA, selectLastTokenKernel, Float16, Float32, BFloat16);

}  // namespace ginfer::core::op::kernel
