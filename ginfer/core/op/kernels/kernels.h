#pragma once

#include <optional>

#include "ginfer/common/device.h"
#include "ginfer/core/tensor/tensor.h"

namespace ginfer::core::op::kernel {

template <typename T, typename Context>
void addKernel(const Context& ctx,
               const tensor::Tensor& a,
               const tensor::Tensor& b,
               tensor::Tensor& c);
using AddKernelFuncType = decltype(&addKernel<float, common::DeviceContext>);

template <typename InT, typename OutT, typename Context>
void argmaxKernel(const Context& ctx, const tensor::Tensor& input, tensor::Tensor& output_idx);
using ArgmaxKernelFuncType = decltype(&argmaxKernel<float, int32_t, common::DeviceContext>);

template <typename T, typename Context>
void embeddingKernel(const Context& ctx,
                     const tensor::Tensor& input,
                     const tensor::Tensor& weight,
                     tensor::Tensor& output);
using EmbeddingKernelFuncType = decltype(&embeddingKernel<float, common::DeviceContext>);

template <typename T, typename Context>
void gemmKernel(const Context& ctx,
                const tensor::Tensor& a,
                const tensor::Tensor& b,
                std::optional<std::reference_wrapper<const tensor::Tensor>> bias,
                tensor::Tensor& c);
using GemmKernelFuncType = decltype(&gemmKernel<float, common::DeviceContext>);

template <typename T, typename Context>
void gemvKernel(const Context& ctx,
                const tensor::Tensor& input,
                const tensor::Tensor& weight,
                std::optional<std::reference_wrapper<const tensor::Tensor>> bias,
                tensor::Tensor& output);
using GemvKernelFuncType = decltype(&gemvKernel<float, common::DeviceContext>);

template <typename T, typename Context>
void GQAKernel(const Context& ctx,
               const tensor::Tensor& q,
               const tensor::Tensor& k,
               const tensor::Tensor& v,
               tensor::Tensor& output);
using GQAKernelFuncType = decltype(&GQAKernel<float, common::DeviceContext>);

template <typename T, typename Context>
void rmsNormKernel(const Context& ctx,
                   const tensor::Tensor& input,
                   const tensor::Tensor& scale,
                   tensor::Tensor& output,
                   float epsilon);
using RMSNormKernelFuncType = decltype(&rmsNormKernel<float, common::DeviceContext>);

template <typename T, typename Context>
void swigluKernel(const Context& ctx,
                  tensor::Tensor& output,
                  const tensor::Tensor& gate,
                  const tensor::Tensor& up);
using SwiGluKernelFuncType = decltype(&swigluKernel<float, common::DeviceContext>);

template <typename T, typename Context>
void selectLastTokenKernel(const Context& ctx,
                           const tensor::Tensor& input,
                           const tensor::Tensor& cu_seqlen_q,
                           tensor::Tensor& output);
using SelectLastTokenKernelFuncType =
    decltype(&selectLastTokenKernel<float, common::DeviceContext>);

}  // namespace ginfer::core::op::kernel
