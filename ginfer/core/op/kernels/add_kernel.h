#pragma once

#include "ginfer/common/device.h"
#include "ginfer/core/tensor/tensor.h"

namespace ginfer::core::op::kernel {

template <typename T, typename Context>
void addKernel(const Context& ctx,
               const tensor::Tensor& a,
               const tensor::Tensor& b,
               tensor::Tensor& c);

using AddKernelFuncType = decltype(&addKernel<float, common::DeviceContext>);
};  // namespace ginfer::core::op::kernel