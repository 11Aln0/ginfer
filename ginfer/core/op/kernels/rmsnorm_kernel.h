#pragma once

#include "ginfer/common/device.h"
#include "ginfer/core/tensor/tensor.h"

namespace ginfer::core::op::kernel {

template <typename T, typename Context>
void rmsNormKernel(const Context& ctx,
                   const tensor::Tensor& input,
                   const tensor::Tensor& scale,
                   tensor::Tensor& output,
                   float epsilon);

using RMSNormKernelFuncType = decltype(&rmsNormKernel<float, common::DeviceContext>);
}  // namespace ginfer::core::op::kernel