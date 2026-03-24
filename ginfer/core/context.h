#pragma once

#include <memory>
#include <optional>
#include "ginfer/common/device.h"
#include "ginfer/core/tensor/tensor.h"

namespace ginfer::core {

struct InferContext {
  std::optional<std::shared_ptr<common::DeviceContext>> dev_ctx;
  std::optional<int> max_seqlen_q;
  std::optional<tensor::TensorRef> cu_seqlens_q;
  std::optional<tensor::TensorRef> cu_seqlens_kv;
  std::optional<tensor::TensorRef> block_tables;
  std::optional<tensor::TensorRef> slot_mapping;

  InferContext() = default;

  InferContext& setMaxSeqlenQ(int max_seqlen_q) {
    this->max_seqlen_q = max_seqlen_q;
    return *this;
  }

  InferContext& setCuSeqlensQ(tensor::TensorRef& cu_seqlens_q) {
    this->cu_seqlens_q = cu_seqlens_q;
    return *this;
  }

  InferContext& setCuSeqlensKV(tensor::TensorRef& cu_seqlens_kv) {
    this->cu_seqlens_kv = cu_seqlens_kv;
    return *this;
  }

  InferContext& setBlockTables(tensor::TensorRef& block_tables) {
    this->block_tables = block_tables;
    return *this;
  }

  InferContext& setSlotMapping(tensor::TensorRef& slot_mapping) {
    this->slot_mapping = slot_mapping;
    return *this;
  }

  InferContext& setDeviceContext(const std::shared_ptr<common::DeviceContext>& dev_ctx) {
    this->dev_ctx = dev_ctx;
    return *this;
  }
};

}  // namespace ginfer::core