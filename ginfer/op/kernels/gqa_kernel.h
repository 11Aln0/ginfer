#pragma once

#include "ginfer/common/device.h"
#include "ginfer/tensor/tensor.h"

namespace ginfer::op::kernel {

template <typename T, typename Context>
void GQAKernel(const Context& ctx, const tensor::Tensor& q, const tensor::Tensor& k,
               const tensor::Tensor& v, tensor::Tensor& output);

using GQAKernelFuncType = decltype(&GQAKernel<float, common::DeviceContext>);

template <typename T, typename Context>
void GQAVarlenKernel(const Context& ctx,
                     const tensor::Tensor& q,
                     const tensor::Tensor& k,
                     const tensor::Tensor& v,
                     const tensor::Tensor& cu_seqlens_q,
                     const tensor::Tensor& cu_seqlens_kv,
                     const tensor::Tensor& block_tables,
                     const int max_seqlen_q,
                     const int paged_block_size,
                     tensor::Tensor& output);

using GQAVarlenKernelFuncType = decltype(&GQAVarlenKernel<float, common::DeviceContext>);

}  // namespace ginfer::op::kernel