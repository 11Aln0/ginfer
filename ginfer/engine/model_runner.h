#pragma once
#include <memory>
#include <vector>
#include "ginfer/core/model/model.h"
#include "ginfer/engine/sequence.h"

namespace ginfer::engine {

class ModelRunner {
 public:
  ModelRunner();
  void prepareBlockTable(std::vector<Sequence::Ptr>& seqs);
  void preparePrefill(std::vector<Sequence::Ptr>& seqs);
  void prepareDecode(std::vector<Sequence::Ptr>& seqs);
  std::vector<int32_t> run(std::vector<Sequence::Ptr>& seqs, bool is_prefill);

 private:
  void allocateKVCache();

 private:
  std::unique_ptr<core::model::Model> model_;
};

}  // namespace ginfer::engine