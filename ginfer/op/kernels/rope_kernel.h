#pragma once

#include "ginfer/common/device.h"
#include "ginfer/tensor/tensor.h"

namespace ginfer::op::kernel {

template <typename T, typename Context>
void CalcSinCosKernel(const Context& ctx, tensor::Tensor sin_cache, tensor::Tensor cos_cache,
                      int start_pos, int end_pos, int head_dim, float rope_theta);

template <typename T, typename Context>
void ROPEKernel(const Context& ctx, const tensor::Tensor input, tensor::Tensor output,
                const tensor::Tensor sin_cache, tensor::Tensor cos_cache);

using CalcSinCosKernelFuncType = decltype(&CalcSinCosKernel<float, common::DeviceContext>);
using ROPEKernelFuncType = decltype(&ROPEKernel<float, common::DeviceContext>);

}  // namespace ginfer::op::kernel
