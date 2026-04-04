#pragma once
#include <tuple>
#include <variant>
#include <vector>
#include "ginfer/engine/config.h"
#include "ginfer/engine/model_runner.h"
#include "ginfer/engine/sampling_params.h"
#include "ginfer/engine/scheduler.h"
#include "ginfer/engine/sequence.h"
#include "ginfer/model/tokenizer/auto_tokenizer.h"

namespace ginfer::engine {

class Engine {
 public:
  Engine(const Config& config);
  std::vector<std::string> generate(
      const std::variant<std::vector<std::string>, std::vector<std::vector<int32_t>>>& prompts,
      const SamplingParams& sampling_params);

 private:
  using Clock = std::chrono::steady_clock;
  struct Stats {
    int total_prefill_tokens = 0;
    int total_decode_tokens = 0;
    std::chrono::microseconds total_prefill_latency{0};
    std::chrono::microseconds total_decode_latency{0};

    void reset() { *this = Stats{}; }
  };

 private:
  void addRequest(const std::string& input, const SamplingParams& sampling_params);
  void addRequest(const std::vector<int32_t>& token_ids, const SamplingParams& sampling_params);
  void updateSeqsPerfStats(const std::vector<Sequence::Ptr>& seqs);
  void updateSchedulePerfStats(int num_tokens, std::chrono::microseconds latency, bool is_prefill);
  void logSeqPerfStats(const Sequence::Ptr& seq);
  void logSchedulePerfStats();

  std::vector<Sequence::Ptr> step();

 private:
  Config cfg_;
  ModelRunner model_runner_;
  Scheduler scheduler_;
  model::tokenizer::AutoTokenizer tokenizer_;
  Stats stats_;
};

}  // namespace ginfer::engine