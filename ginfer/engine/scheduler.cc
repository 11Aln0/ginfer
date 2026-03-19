#include "ginfer/engine/scheduler.h"
#include <glog/logging.h>
#include <ranges>

namespace ginfer::engine {

bool Scheduler::is_finished() { return waiting.empty() && running.empty(); }

void Scheduler::add(Sequence::Ptr& seq) { waiting.push(seq); }

std::tuple<std::vector<Sequence::Ptr>, bool> Scheduler::schedule() {
  std::vector<Sequence::Ptr> scheduled_seqs;
  int num_seqs = 0, num_batched_tokens = 0;

  // prefill first
  while (!waiting.empty() && num_seqs < max_num_seqs) {
    auto& seq = waiting.front();
    if (num_batched_tokens + seq->num_tokens > max_num_batched_tokens ||
        !block_mgr.canAllocate(seq))
      break;
    num_seqs++;
    num_batched_tokens += seq->num_tokens - seq->num_cached_tokens;
    seq->state = SequenceState::kRunning;
    block_mgr.allocate(seq);
    waiting.pop();
    running.push_back(seq);
    scheduled_seqs.push_back(seq);
  }

  if (!scheduled_seqs.empty()) return {std::move(scheduled_seqs), true};

  // decode

  while (!running.empty() && num_seqs < max_num_seqs) {
    auto& seq = running.front();
    running.pop_front();
    while (!block_mgr.canAppend(seq)) {
      if (!running.empty()) {
        auto& s = running.back();
        running.pop_back();
        preempt(s);
      } else {
        preempt(seq);
        break;
      }
    }

    if (seq->state == SequenceState::kRunning) {
      // seq is not preempted
      block_mgr.append(seq);
      num_seqs++;
      scheduled_seqs.push_back(seq);
    }
  }

  for (auto& seq : scheduled_seqs | std::views::reverse) {
    running.push_front(seq);
  }

  return {std::move(scheduled_seqs), false};
}

void Scheduler::preempt(Sequence::Ptr& seq) {
  block_mgr.release(seq);
  seq->state = SequenceState::kWaiting;
  waiting.push(seq);
}

void Scheduler::postprocess(std::vector<Sequence::Ptr>& seqs,
                            const std::vector<int32_t>& token_ids) {
  CHECK(seqs.size() == token_ids.size())
      << "The number of sequences and token ids must be the same";
  for (size_t i = 0; i < seqs.size(); i++) {
    auto& seq = seqs[i];
    int32_t token_id = token_ids[i];
    if (token_id == eos) {
      block_mgr.release(seq);
      seq->state = SequenceState::kFinished;
      running.erase(std::find(running.begin(), running.end(),
                              [&seq](const auto& s) { return s->seq_id == seq->seq_id; }));
    }
  }
}

}  // namespace ginfer::engine