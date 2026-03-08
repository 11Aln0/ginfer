#pragma once

#include "ginfer/common/device.h"
#include "ginfer/core/tensor/tensor.h"

namespace ginfer::core::op::kernel {

template <typename T, typename Context>
void argmaxKernel(const Context& ctx, const tensor::Tensor& input, tensor::Tensor& output_idx);

using ArgmaxKernelFuncType = decltype(&argmaxKernel<float, common::DeviceContext>);

}  // namespace ginfer::core::op::kernel
