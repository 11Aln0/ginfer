#pragma once

#include "ginfer/common/device.h"
#include "ginfer/tensor/tensor.h"
#include <optional>


namespace ginfer::op::kernel {

template <typename T, typename Context>
void gemvKernel(const Context& ctx, const tensor::Tensor& input, const tensor::Tensor& weight, std::optional<std::reference_wrapper<const tensor::Tensor>> bias, tensor::Tensor& output);
using GemvKernelFuncType = decltype(&gemvKernel<float, common::DeviceContext>);

}  // namespace ginfer::op::kernel