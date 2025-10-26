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
                           int n) {

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
  int n = static_cast<int>(a.size());
  const T* a_data = a.data<T>();
  const T* b_data = b.data<T>();
  T* c_data = c.data<T>();

  int blockSize = 256;
  int numBlocks = (n + blockSize - 1) / blockSize;
  if(ctx.stream() == nullptr) {
    addKernelImpl<T><<<numBlocks, blockSize>>>(a_data, b_data, c_data, n);
  } else {
    addKernelImpl<T><<<numBlocks, blockSize, 0, ctx.stream()>>>(a_data, b_data, c_data, n);
  }
  
}

REGISTER_KERNEL(add, 
                common::DeviceType::kDeviceCUDA, 
                addKernel, 
                tensor::DType::kDTypeFloat32, 
                tensor::DType::kDTypeFloat16);

}  // namespace ginfer::op::kernel