#pragma once

#include "ginfer/common/device.h"
#include "ginfer/tensor/tensor.h"

namespace ginfer::op::kernel {

template <typename T, typename Context>
void RotaryEmbeddingKernel(const Context& ctx, tensor::Tensor sin_cache, tensor::Tensor cos_cache, int start_pos,
                           int end_pos, float rope_theta);

template <typename T, typename Context>
void ROPEKernel(const Context& ctx, const tensor::Tensor input, tensor::Tensor output, const tensor::Tensor sin_cache,
                tensor::Tensor cos_cache);

using RotaryEmbeddingKernelFuncType = decltype(&RotaryEmbeddingKernel<float, common::DeviceContext>);
using ROPEKernelFuncType = decltype(&ROPEKernel<float, common::DeviceContext>);

}  // namespace ginfer::op::kernel
