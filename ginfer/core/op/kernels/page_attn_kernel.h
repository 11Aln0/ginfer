#pragma once

#include "ginfer/common/device.h"
#include "ginfer/core/tensor/tensor.h"

namespace ginfer::core::op::kernel {

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

template <typename T, typename Context>
void storeKVCacheKernel(const Context& ctx,
                        const tensor::Tensor& k,
                        const tensor::Tensor& v,
                        tensor::Tensor& k_cache,
                        tensor::Tensor& v_cache,
                        const tensor::Tensor& slot_mapping);

using StoreKVCacheKernelFuncType = decltype(&storeKVCacheKernel<float, common::DeviceContext>);

}  // namespace ginfer::core::op::kernel