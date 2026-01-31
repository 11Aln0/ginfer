#include "hgemm_NT.cuh"
#include "ginfer/op/kernels/cuda/intrinsic.cuh"
#include "ginfer/tensor/tensor.h"
#include "ginfer/common/device.h"
#include "ginfer/op/kernels/kernel_registry.h"

namespace ginfer::op::kernel {


template <typename T, typename Context>
void gemmKernel(const Context& ctx,
                const tensor::Tensor& a,
                const tensor::Tensor& b,
                tensor::Tensor& c) {

  CHECK(ctx.getDeviceType() == common::DeviceType::kDeviceCUDA)
      << "gemvKernel only supports CUDA device type.";
  CHECK(b.layout() == tensor::Layout::kLayoutColMajor)
      << "gemvKernel only supports col-major matrix layout.";

  auto cuda_ctx = static_cast<const common::CUDADeviceContext&>(ctx);
  
  const T* a_data = a.data<T>();
  const T* b_data = b.data<T>();
  T* c_data = c.data<T>();

  const size_t M = a.shape()[0];
  const size_t K = a.shape()[1];
  const size_t N = b.shape()[1];

  int block_dim = 256;
  constexpr int BM = 128, BN = 128;
  dim3 grid_dim((N + BN - 1) / BN, (M + BM - 1) / BM);

  mma2x4_warp4x4_bce_swizzle_stagen_hgemm_kernel<BM, BN, 16, 2><<<grid_dim, block_dim, 0, cuda_ctx.getStream()>>>(
      M, N, K, a_data, b_data, c_data);

}

REGISTER_KERNEL(gemm,
                kDeviceCUDA,
                gemmKernel,
                tensor::DataType::kDataTypeFloat16);  

} // namespace ginfer::op::kernel