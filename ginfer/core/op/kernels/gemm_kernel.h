#pragma once

#include <optional>
#include "ginfer/common/device.h"
#include "ginfer/core/tensor/tensor.h"

namespace ginfer::core::op::kernel {

template <typename T, typename Context>
void gemmKernel(const Context& ctx,
                const tensor::Tensor& a,
                const tensor::Tensor& b,
                std::optional<std::reference_wrapper<const tensor::Tensor>> bias,
                tensor::Tensor& c);
using GemmKernelFuncType = decltype(&gemmKernel<float, common::DeviceContext>);

}  // namespace ginfer::core::op::kernel