#pragma once

#include "ginfer/common/device.h"
#include "ginfer/tensor/tensor.h"

namespace ginfer::op::kernel {

template <typename T, typename Context>
void swigluKernel(const Context& ctx, tensor::Tensor& output, const tensor::Tensor& gate, const tensor::Tensor& up);

using SwiGluKernelFuncType = decltype(&swigluKernel<float, common::DeviceContext>);

}  // namespace ginfer::op::kernel
