#include "ginfer/op/kernels/kernel_registry.h"
#include "ginfer/common/check.h"
#include "ginfer/op/kernels/cuda/vectorize.cuh"
#include "ginfer/op/kernels/add_kernel.h"

#include <glog/logging.h>

namespace ginfer::op::kernel {


template<typename T, int vec_size = DefaultVecSize<T>::value>
__global__ void addKernelImpl(const T* __restrict__ a, 
                              const T* __restrict__ b, 
                              T* __restrict__ c, 
                              size_t n) {
                                
  using AccessT = AlignedVector<T, vec_size>;

  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = (size_t)gridDim.x * blockDim.x;

  size_t vec_n = n / vec_size;
  const AccessT* a_vec = reinterpret_cast<const AccessT*>(a);
  const AccessT* b_vec = reinterpret_cast<const AccessT*>(b);
  AccessT* c_vec = reinterpret_cast<AccessT*>(c);

  for(size_t i = tid; i < vec_n; i += stride) {
    AccessT va = a_vec[i];
    AccessT vb = b_vec[i];
    AccessT vc = va + vb;
    c_vec[i] = vc;
  }

  size_t offset = vec_n * vec_size;
  for(size_t i = offset + tid; i < n; i += stride) {
    c[i] = a[i] + b[i];
  }
}

template<typename T, typename Context>
void addKernel(const Context& ctx, const tensor::Tensor& a, const tensor::Tensor& b, tensor::Tensor& c) {
  CHECK(ctx.getDeviceType() == common::DeviceType::kDeviceCUDA)
      << "addKernel only supports CUDA device type.";
  
  auto cuda_ctx = static_cast<const common::CUDADeviceContext&>(ctx);
  size_t n = a.size();
  const T* a_data = a.data<T>();
  const T* b_data = b.data<T>();
  T* c_data = c.data<T>();

  int block_dim = 256;
  int grid_dim = (n + block_dim - 1) / block_dim;
  addKernelImpl<T><<<grid_dim, block_dim, 0, cuda_ctx.getStream()>>>(a_data, b_data, c_data, n);

  // cudaDeviceSynchronize();
  // cudaError_t err = cudaGetLastError();
  // CHECK(err == cudaSuccess) << "CUDA kernel launch failed: " << cudaGetErrorString(err);
  
}

REGISTER_KERNEL(add, 
                kDeviceCUDA, 
                addKernel,
                tensor::DataType::kDataTypeInt32,
                tensor::DataType::kDataTypeFloat32, 
                tensor::DataType::kDataTypeFloat16,
                tensor::DataType::kDataTypeBFloat16);


}  // namespace ginfer::op::kernel