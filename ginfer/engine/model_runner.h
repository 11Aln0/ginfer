#pragma once
#include <memory>
#include <vector>
#include "ginfer/core/context.h"
#include "ginfer/core/memory/allocator.h"
#include "ginfer/core/model/model.h"
#include "ginfer/core/tensor/tensor.h"
#include "ginfer/engine/config.h"
#include "ginfer/engine/sequence.h"

namespace ginfer::engine {

class ModelRunner {
 public:
  ModelRunner(Config config);
  std::tuple<core::tensor::TensorRef, core::tensor::TensorRef> prepareDecode(
      core::InferContext& ctx, std::vector<Sequence::Ptr>& seqs);
  std::tuple<core::tensor::TensorRef, core::tensor::TensorRef> preparePrefill(
      core::InferContext& ctx, std::vector<Sequence::Ptr>& seqs);
  Result<std::vector<int32_t>, std::string> run(std::vector<Sequence::Ptr>& seqs, bool is_prefill);

 private:
  using DeviceType = common::DeviceType;
  using Model = core::model::Model;
  using Tensor = core::tensor::Tensor;
  using TensorRef = core::tensor::TensorRef;

 private:
  struct Workspace {
    TensorRef input_ids;
    TensorRef positions;
    TensorRef cu_seqlens_q;
    TensorRef cu_seqlens_kv;
    TensorRef slot_mapping;
    TensorRef block_tables;
  };

  void allocateKVCache();
  void allocateWorkspace();
  void resetContext(core::InferContext& ctx);

  int getMaxBlocksPerSeq() const;

  void loadModel();
  void warmupModel();

  void prepareBlockTables(std::vector<Sequence::Ptr>& seqs, TensorRef& block_tables_host);
  Workspace prepareWorkspace(Workspace& from, int total_q_tokens, int batch_size);

 private:
  std::unique_ptr<core::model::Model> model_;
  size_t num_kvcache_blocks_;
  Config config_;
  core::memory::DeviceAllocator* allocator_;
  core::memory::DeviceAllocator* pooled_allocator_;
  Workspace host_workspace_;
  Workspace dev_workspace_;
};

}  // namespace ginfer::engine