#pragma once

#include "ginfer/common/device.h"
#include "ginfer/core/tensor/tensor.h"

namespace ginfer::core::op::kernel {

template <typename T, typename Context>
void GQAKernel(const Context& ctx,
               const tensor::Tensor& q,
               const tensor::Tensor& k,
               const tensor::Tensor& v,
               tensor::Tensor& output);

using GQAKernelFuncType = decltype(&GQAKernel<float, common::DeviceContext>);

}  // namespace ginfer::core::op::kernel