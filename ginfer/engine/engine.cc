#include "ginfer/engine/engine.h"
#include <chrono>
#include <map>

namespace ginfer::engine {

namespace {

using Clock = std::chrono::steady_clock;

bool isUnset(const Sequence::TimePoint& ts) { return ts == Sequence::TimePoint{}; }

void logSequenceStats(const Sequence::Ptr& seq) {
  auto output_token_count = seq->num_tokens - seq->num_prompt_tokens;
  auto e2e_latency_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(seq->finish_ts - seq->req_ts);
  auto ttft_us =
      std::chrono::duration_cast<std::chrono::microseconds>(seq->first_token_ts - seq->req_ts)
          .count();
  auto decode_latency_us =
      std::chrono::duration_cast<std::chrono::microseconds>(seq->finish_ts - seq->first_token_ts);
  auto tpot_us = decode_latency_us.count() / (output_token_count - 1);

  LOG(INFO) << "seq_id=" << seq->seq_id << " prompt_tokens=" << seq->num_prompt_tokens
            << " output_tokens=" << output_token_count << " TTFT=" << (float)ttft_us / 1000.0
            << " ms"
            << " TOPT=" << (float)tpot_us / 1000.0 << " ms"
            << " e2e_latency=" << e2e_latency_ms.count() << " ms";
}

}  // namespace

Engine::Engine(const Config& config)
    : cfg_(config),
      model_runner_(config),
      scheduler_(config.max_num_seqs,
                 config.max_num_batched_tokens,
                 config.model_config.eos_token_ids,
                 BlockManager(model_runner_.getNumKVCacheBlocks(), config.kvcache_block_size)),
      tokenizer_(config.model_path) {}

void Engine::addRequest(const std::string& input) {
  auto conversation = nlohmann::json::array({{{"role", "user"}, {"content", input}}});
  auto token_ids = tokenizer_.encode(tokenizer_.applyChatTemplate(conversation));
  auto seq = Sequence::create(std::move(token_ids), cfg_.kvcache_block_size);
  seq->req_ts = Clock::now();
  scheduler_.add(seq);
}

void Engine::updateSequenceStats(const std::vector<Sequence::Ptr>& seqs) {
  auto now = Clock::now();
  for (const auto& seq : seqs) {
    if (isUnset(seq->first_token_ts) && seq->num_tokens > seq->num_prompt_tokens) {
      seq->first_token_ts = now;
    }
    if (seq->state == SequenceState::kFinished) {
      seq->finish_ts = now;
    }
  }
}

std::vector<Sequence::Ptr> Engine::step() {
  auto [seqs, is_prefill] = scheduler_.schedule();
  if (seqs.empty()) return {};

  auto result = model_runner_.run(seqs, is_prefill);
  if (!result.ok()) {
    LOG(ERROR) << "Inference error: " << result.err();
    return {};
  }

  auto token_ids = result.value();
  scheduler_.postprocess(seqs, token_ids);
  updateSequenceStats(seqs);

  std::erase_if(seqs,
                [](const Sequence::Ptr& seq) { return seq->state != SequenceState::kFinished; });

  return seqs;
}

std::vector<std::string> Engine::generate(const std::vector<std::string>& prompts) {
  for (const auto& prompt : prompts) {
    addRequest(prompt);
  }
  std::map<int, std::string> outputs_m;
  while (!scheduler_.is_finished()) {
    auto seqs = step();
    for (const auto& seq : seqs) {
      logSequenceStats(seq);
      outputs_m[seq->seq_id] = tokenizer_.decode(seq->token_ids, true);
    }
  }

  std::vector<std::string> outputs;
  outputs.reserve(outputs_m.size());
  for (auto& [_, output] : outputs_m) {
    outputs.push_back(std::move(output));
  }
  return outputs;
}

}  // namespace ginfer::engine