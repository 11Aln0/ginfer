#include "ginfer/engine/block_manager.h"
#include <glog/logging.h>
#include <xxhash.h>
#include "ginfer/common/errors.h"

namespace ginfer::engine {

BlockManager::BlockManager(int block_size) {
  block_size_ = block_size;
  blocks_.reserve(block_size);
  for (int i = 0; i < block_size_; i++) {
    blocks_.emplace_back(i);
    free_block_ids_.push_back(i);
  }
}

int BlockManager::getBlockId(std::optional<uint64_t> hash) {
  if (hash.has_value()) {
    auto it = hash_to_block_id_.find(hash.value());
    if (it != hash_to_block_id_.end()) {
      return it->second;
    }
  }
  return -1;
}

uint64_t BlockManager::computeHash(std::span<int32_t> token_ids, std::optional<uint64_t> prefix) {
  XXH64_state_t* state = XXH64_createState();
  XXH64_reset(state, 0);
  if (prefix.has_value()) {
    uint64_t prefix_hash = prefix.value();
    XXH64_update(state, &prefix_hash, sizeof(prefix_hash));
  }
  XXH64_update(state, token_ids.data(), token_ids.size() * sizeof(int32_t));
  uint64_t hash = XXH64_digest(state);
  XXH64_freeState(state);
  return hash;
}

Block& BlockManager::allocateBlock(int block_id) {
  auto& blk = blocks_[block_id];
  blk.reset();
  auto it = std::find(free_block_ids_.begin(), free_block_ids_.end(), block_id);
  if (it != free_block_ids_.end()) {
    free_block_ids_.erase(it);
  } else {
    LOG(WARNING) << "Block " << block_id << " is already removed";
  }
  used_block_ids_.insert(block_id);
  return blk;
}

void BlockManager::releaseBlock(int block_id) {
  auto blk = blocks_[block_id];
  blk.ref_cnt--;
  if (blk.ref_cnt == 0) {
    used_block_ids_.erase(block_id);
    free_block_ids_.push_back(block_id);
  }
}

bool BlockManager::canAllocate(Sequence::Ptr& seq) const {
  return free_block_ids_.size() >= seq->numBlocks();
}

void BlockManager::allocate(Sequence::Ptr& seq) {
  CHECK(seq->block_table.empty()) << "Sequence already has blocks allocated";

  std::optional<uint64_t> h = std::nullopt;
  bool cache_miss = false;

  int n_blk = seq->numBlocks();
  for (int i = 0; i < n_blk; i++) {
    auto token_ids = seq->getBlock(i);

    // compute hash only when block is full
    h = token_ids.size() == block_size_ ? std::make_optional(computeHash(token_ids, h))
                                        : std::nullopt;

    int block_id = getBlockId(h);

    if (block_id == -1 ||
        !std::equal(token_ids.begin(), token_ids.end(), blocks_[block_id].token_ids.begin())) {
      cache_miss = true;
    }

    if (cache_miss) {
      block_id = free_block_ids_.front();
      allocateBlock(block_id);
    } else {
      seq->num_cached_tokens += block_size_;
      if (used_block_ids_.find(block_id) != used_block_ids_.end()) {
        blocks_[block_id].ref_cnt++;
      } else {
        used_block_ids_.insert(block_id);
        allocateBlock(block_id);
      }
    }

    // cache only when block is full
    if (h.has_value()) {
      auto hash = h.value();
      blocks_[block_id].update(hash, token_ids);
      hash_to_block_id_[hash] = block_id;
    }
    seq->block_table.push_back(block_id);
  }
}

void BlockManager::release(Sequence::Ptr& seq) {
  auto begin = seq->block_table.begin(), end = seq->block_table.end();
  for (auto it = begin; it != end; it++) {
    releaseBlock(*it);
  }
  seq->num_cached_tokens = 0;
  seq->block_table.clear();
}

bool BlockManager::canAppend(Sequence::Ptr& seq) const {
  if (seq->num_tokens % block_size_ == 1) {
    // need to allocate a new block for the last token
    return free_block_ids_.size() > 0;
  } else {
    return true;
  }
}

void BlockManager::append(Sequence::Ptr& seq) {
  if (seq->num_tokens % block_size_ == 1) {
    // need to allocate a new block for the last token
    int block_id = free_block_ids_.front();
    allocateBlock(block_id);
    seq->block_table.push_back(block_id);
  } else if (seq->num_tokens % block_size_ == 0) {
    // need to update the hash of the last block
    int n_blk = seq->numBlocks();
    int last_block_id = seq->block_table[n_blk - 1];
    auto token_ids = seq->getBlock(n_blk - 1);

    auto prefix = n_blk > 1 ? blocks_[seq->block_table[n_blk - 2]].hash : std::nullopt;
    auto hash = computeHash(token_ids, prefix);

    blocks_[last_block_id].update(hash, token_ids);  // TODO: only insert token_id to back
    hash_to_block_id_[hash] = last_block_id;
  }
  seq->num_tokens++;
}

};  // namespace ginfer::engine