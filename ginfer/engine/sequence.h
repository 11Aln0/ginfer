#pragma once
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace ginfer::engine {

enum class SequenceState {
  kWaiting,
  kRunning,
  kFinished,
};

struct Sequence {
  using Ptr = std::shared_ptr<Sequence>;

  std::vector<int32_t> token_ids;
  SequenceState state;

  int seq_id;
  int num_tokens;
  int num_prompt_tokens;
  int num_cached_tokens;
  const int block_size;
  std::vector<int> block_table;

  Sequence(int seq_id, std::vector<int32_t> token_ids, const int block_size)
      : seq_id(seq_id), block_size(block_size) {
    this->token_ids = std::move(token_ids);
    this->state = SequenceState::kWaiting;
    this->num_tokens = token_ids.size();
    this->num_prompt_tokens = this->num_tokens;
    this->num_cached_tokens = 0;
  }

  int numBlocks() const { return (num_tokens + block_size - 1) / block_size; }

  std::span<int32_t> getBlock(int block_id) {
    int start = block_id * block_size;
    int end = std::min(start + block_size, num_tokens);
    return std::span<int32_t>(token_ids).subspan(start, end - start);
  }

  void appendToken(int32_t token_id) {
    token_ids.push_back(token_id);
    num_tokens++;
  }
};

};  // namespace ginfer::engine
