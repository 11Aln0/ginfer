#pragma once
#include <vector>
#include "ginfer/engine/config.h"
#include "ginfer/engine/model_runner.h"
#include "ginfer/engine/scheduler.h"
#include "ginfer/engine/sequence.h"
#include "ginfer/model/tokenizer/auto_tokenizer.h"

namespace ginfer::engine {

class Engine {
 public:
  Engine(const Config& config);
  std::vector<Sequence::Ptr> step();
  std::vector<std::string> generate(const std::vector<std::string>& prompts);

 private:
  void addRequest(const std::string& input);
  void updateSequenceStats(const std::vector<Sequence::Ptr>& seqs);

 private:
  Config cfg_;
  ModelRunner model_runner_;
  Scheduler scheduler_;
  model::tokenizer::AutoTokenizer tokenizer_;
};

}  // namespace ginfer::engine