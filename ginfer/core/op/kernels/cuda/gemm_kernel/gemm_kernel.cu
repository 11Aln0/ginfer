#include "ginfer/common/device.h"
#include "ginfer/core/op/kernels/cuda/intrinsic.cuh"
#include "ginfer/core/op/kernels/gemm_kernel.h"
#include "ginfer/core/op/kernels/kernel_registry.h"
#include "ginfer/core/tensor/tensor.h"
#include "hgemm_NT.cuh"

namespace ginfer::core::op::kernel {

template <typename T, typename Context>
void gemmKernel(const Context& ctx,
                const tensor::Tensor& a,
                const tensor::Tensor& b,
                std::optional<std::reference_wrapper<const tensor::Tensor>> bias,
                tensor::Tensor& c) {
  CHECK(ctx.getDeviceType() == common::DeviceType::kDeviceCUDA)
      << "gemvKernel only supports CUDA device type.";

  auto cuda_ctx = static_cast<const common::CUDADeviceContext&>(ctx);

  const auto& a_shape = a.shape();
  const auto& b_shape = b.shape();
  const auto& b_strides = b.strides();
  CHECK(a_shape.ndim() == 2 && b_shape.ndim() == 2) << "gemmKernel only supports 2D tensors.";
  CHECK(a_shape[1] == b_shape[0]) << "Inner dimensions of A and B must match.";
  CHECK(b_strides[1] == b_shape[0]) << "B must be in column-major order.";
  const size_t M = a_shape[0];
  const size_t K = a_shape[1];
  const size_t N = b_shape[1];

  const T* a_data = a.data<T>();
  const T* b_data = b.data<T>();
  T* c_data = c.data<T>();

  int block_dim = 256;
  constexpr int BM = 128, BN = 128;
  dim3 grid_dim((N + BN - 1) / BN, (M + BM - 1) / BM);

  if (bias.has_value()) {
    const T* bias_data = bias->get().data<T>();
    mma2x4_warp4x4_bce_swizzle_stagen_hgemm_kernel<T, true, BM, BN>
        <<<grid_dim, block_dim, 0, cuda_ctx.getStream()>>>(M, N, K, a_data, b_data, bias_data,
                                                           c_data);
  } else {
    mma2x4_warp4x4_bce_swizzle_stagen_hgemm_kernel<T, false, BM, BN>
        <<<grid_dim, block_dim, 0, cuda_ctx.getStream()>>>(M, N, K, a_data, b_data, nullptr,
                                                           c_data);
  }
}

REGISTER_KERNEL(gemm,
                kDeviceCUDA,
                gemmKernel,
                tensor::DataType::kDataTypeFloat16,
                tensor::DataType::kDataTypeBFloat16);

}  // namespace ginfer::core::op::kernel