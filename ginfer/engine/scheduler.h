#pragma once
#include <memory>
#include <queue>
#include "ginfer/engine/block_manager.h"
#include "ginfer/engine/sequence.h"

namespace ginfer::engine {

class Scheduler {
 public:
  bool is_finished();
  void add(Sequence::Ptr& seq);
  std::tuple<std::vector<Sequence::Ptr>, bool> schedule();
  void postprocess(std::vector<Sequence::Ptr>& seqs, const std::vector<int32_t>& token_ids);

 private:
  void preempt(Sequence::Ptr& seq);

 private:
  int max_num_seqs;
  int max_num_batched_tokens;
  int32_t eos;
  BlockManager block_mgr;
  std::queue<Sequence::Ptr> waiting;
  std::deque<Sequence::Ptr> running;
};

}  // namespace ginfer::engine