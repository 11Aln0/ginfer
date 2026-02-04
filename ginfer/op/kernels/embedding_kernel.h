#pragma once

#include "ginfer/common/device.h"
#include "ginfer/tensor/tensor.h"

namespace ginfer::op::kernel {

template <typename T, typename Context>
void embeddingKernel(const Context& ctx, const tensor::Tensor& input,
                     const tensor::Tensor& weight, tensor::Tensor& output);

using EmbeddingKernelFuncType = decltype(&embeddingKernel<float, common::DeviceContext>);

}  // namespace ginfer::op::kernel
