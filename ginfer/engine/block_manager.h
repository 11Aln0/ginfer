#pragma once
#include <cstdint>
#include <optional>
#include <queue>
#include <span>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "ginfer/engine/sequence.h"

namespace ginfer::engine {

struct Block {
  int block_id;
  int ref_cnt;
  std::optional<uint64_t> hash;
  std::vector<int32_t> token_ids;

  Block(int block_id) {
    this->block_id = block_id;
    this->ref_cnt = 0;
    this->hash = std::nullopt;
  }

  void update(std::optional<uint64_t> hash, std::span<const int32_t> token_ids) {
    this->hash = hash;
    this->token_ids.assign(token_ids.begin(), token_ids.end());
  }

  void reset() {
    this->ref_cnt = 1;
    this->hash = std::nullopt;
    this->token_ids.resize(0);
  }
};

class BlockManager {
 public:
  BlockManager(int num_blocks, int block_size);

  bool canAllocate(Sequence::Ptr& seq) const;
  void allocate(Sequence::Ptr& seq);
  void release(Sequence::Ptr& seq);

  bool canAppend(Sequence::Ptr& seq) const;
  void append(Sequence::Ptr& seq);

 private:
  static uint64_t computeHash(std::span<int32_t> token_ids,
                              std::optional<uint64_t> prefix = std::nullopt);
  int getBlockId(std::optional<uint64_t> hash);

  Block& allocateBlock(int block_id);
  void releaseBlock(int block_id);

 private:
  int block_size_;
  int num_blocks_;
  std::vector<Block> blocks_;
  std::deque<int> free_block_ids_;
  std::unordered_set<int> used_block_ids_;
  std::unordered_map<uint64_t, int> hash_to_block_id_;
};

};  // namespace ginfer::engine