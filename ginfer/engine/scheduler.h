#pragma once
#include <memory>
#include <queue>
#include <vector>
#include "ginfer/engine/block_manager.h"
#include "ginfer/engine/config.h"
#include "ginfer/engine/sequence.h"

namespace ginfer::engine {

class Scheduler {
 public:
  Scheduler(int max_num_seqs,
            int max_num_batched_tokens,
            std::vector<int32_t> eos,
            BlockManager block_mgr);
  bool is_finished();
  void add(Sequence::Ptr& seq);
  std::tuple<std::vector<Sequence::Ptr>, bool> schedule();
  void postprocess(std::vector<Sequence::Ptr>& seqs, const std::vector<int32_t>& token_ids);

 private:
  void preempt(Sequence::Ptr& seq);

  bool isEosToken(int32_t token_id) const;

 private:
  int max_num_seqs;
  int max_num_batched_tokens;
  std::vector<int32_t> eos;
  BlockManager block_mgr;
  std::queue<Sequence::Ptr> waiting;
  std::deque<Sequence::Ptr> running;
};

}  // namespace ginfer::engine