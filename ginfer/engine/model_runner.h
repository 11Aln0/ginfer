#pragma once
#include <memory>
#include <vector>
#include "ginfer/core/memory/allocator.h"
#include "ginfer/core/model/model.h"
#include "ginfer/core/tensor/tensor.h"
#include "ginfer/engine/config.h"
#include "ginfer/engine/sequence.h"

namespace ginfer::engine {

class ModelRunner {
 public:
  ModelRunner(Config config);
  void prepareBlockTables(std::vector<Sequence::Ptr>& seqs);
  void preparePrefill(std::vector<Sequence::Ptr>& seqs);
  void prepareDecode(std::vector<Sequence::Ptr>& seqs);
  std::vector<int32_t> run(std::vector<Sequence::Ptr>& seqs, bool is_prefill);

 private:
  using DeviceType = common::DeviceType;
  using Model = core::model::Model;
  using Tensor = core::tensor::Tensor;
  using TensorRef = core::tensor::TensorRef;

 private:
  void allocateKVCache();
  void loadModel();
  void warmupModel();

 private:
  std::unique_ptr<core::model::Model> model_;
  size_t num_kvcache_blocks_;
  Config config_;
  core::memory::DeviceAllocator* allocator_;
  core::memory::DeviceAllocator* pooled_allocator_;
};

}  // namespace ginfer::engine