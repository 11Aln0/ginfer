#pragma once

#include "ginfer/common/device.h"
#include "ginfer/tensor/tensor.h"

namespace ginfer::op::kernel {

template <typename T, typename Context>
void gemmKernel(const Context& ctx, const tensor::Tensor& a, const tensor::Tensor& b, tensor::Tensor& c);
using GemmKernelFuncType = decltype(&gemmKernel<float, common::DeviceContext>);

}  // namespace ginfer::op::kernel