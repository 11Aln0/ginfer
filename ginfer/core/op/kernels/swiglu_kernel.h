#pragma once

#include "ginfer/common/device.h"
#include "ginfer/core/tensor/tensor.h"

namespace ginfer::core::op::kernel {

template <typename T, typename Context>
void swigluKernel(const Context& ctx,
                  tensor::Tensor& output,
                  const tensor::Tensor& gate,
                  const tensor::Tensor& up);

using SwiGluKernelFuncType = decltype(&swigluKernel<float, common::DeviceContext>);

}  // namespace ginfer::core::op::kernel
