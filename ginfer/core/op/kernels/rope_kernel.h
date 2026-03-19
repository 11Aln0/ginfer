#pragma once

#include "ginfer/common/device.h"
#include "ginfer/core/tensor/tensor.h"

namespace ginfer::core::op::kernel {

template <typename T, typename Context>
void RotaryEmbeddingKernel(const Context& ctx,
                           tensor::Tensor& sin_cache,
                           tensor::Tensor& cos_cache,
                           int start_pos,
                           int end_pos,
                           float rope_theta);

template <typename T, typename Context>
void Llama3RotaryEmbeddingKernel(const Context& ctx,
                                 tensor::Tensor& sin_cache,
                                 tensor::Tensor& cos_cache,
                                 int start_pos,
                                 int end_pos,
                                 float rope_theta,
                                 float factor,
                                 float high_freq_factor,
                                 float low_freq_factor,
                                 int old_ctx_len);

template <typename T, typename Context>
void ROPEKernel(const Context& ctx,
                const tensor::Tensor& input,
                const tensor::Tensor& positions,
                const tensor::Tensor& sin_cache,
                const tensor::Tensor& cos_cache,
                tensor::Tensor& output);

using Llama3RotaryEmbeddingKernelFuncType =
    decltype(&Llama3RotaryEmbeddingKernel<float, common::DeviceContext>);
using RotaryEmbeddingKernelFuncType =
    decltype(&RotaryEmbeddingKernel<float, common::DeviceContext>);
using ROPEKernelFuncType = decltype(&ROPEKernel<float, common::DeviceContext>);

}  // namespace ginfer::core::op::kernel
