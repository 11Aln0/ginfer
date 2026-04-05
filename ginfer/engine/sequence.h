#pragma once
#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>
#include "ginfer/engine/sampling_params.h"

namespace ginfer::engine {

enum class SequenceState {
  kWaiting,
  kRunning,
  kFinished,
};

struct Sequence {
  using Ptr = std::shared_ptr<Sequence>;
  using TimePoint = std::chrono::steady_clock::time_point;

  std::vector<int32_t> token_ids;
  SequenceState state;

  int seq_id;
  int num_tokens;
  int num_prompt_tokens;
  int num_cached_tokens;
  int max_tokens;
  bool ignore_eos;

  const int block_size;
  std::vector<int> block_table;

  TimePoint req_ts;
  TimePoint first_token_ts;
  TimePoint finish_ts;

  static Sequence::Ptr create(TimePoint req_ts,
                              std::vector<int32_t> token_ids,
                              int block_size,
                              const SamplingParams& sampling_params) {
    return Sequence::Ptr(new Sequence(counter.fetch_add(1, std::memory_order_relaxed), req_ts,
                                      std::move(token_ids), block_size, sampling_params));
  }

  int numBlocks() const { return (num_tokens + block_size - 1) / block_size; }

  int numCachedBlocks() const { return num_cached_tokens / block_size; }

  std::span<int32_t> getBlock(int block_id) {
    int start = block_id * block_size;
    int end = std::min(start + block_size, num_tokens);
    return std::span<int32_t>(token_ids).subspan(start, end - start);
  }

  void appendToken(int32_t token_id) {
    token_ids.push_back(token_id);
    num_tokens++;
  }

 private:
  Sequence(int seq_id,
           TimePoint req_ts,
           std::vector<int32_t> token_ids,
           const int block_size,
           const SamplingParams& sampling_params)
      : req_ts(req_ts), seq_id(seq_id), block_size(block_size) {
    this->num_tokens = token_ids.size();
    this->token_ids = std::move(token_ids);
    this->state = SequenceState::kWaiting;
    this->num_prompt_tokens = this->num_tokens;
    this->num_cached_tokens = 0;
    this->max_tokens = sampling_params.max_tokens;
    this->ignore_eos = sampling_params.ignore_eos;
  }

 private:
  static inline std::atomic<int> counter{0};
};

};  // namespace ginfer::engine
