#include "ginfer/engine/engine.h"
#include <chrono>
#include <cmath>
#include <map>
#include "ginfer/utils/variant.h"

namespace ginfer::engine {

Engine::Engine(const Config& config)
    : cfg_(config),
      model_runner_(config),
      scheduler_(config.max_num_seqs,
                 config.max_num_batched_tokens,
                 config.model_config.eos_token_ids,
                 BlockManager(model_runner_.getNumKVCacheBlocks(), config.kvcache_block_size)),
      tokenizer_(config.model_path) {}

void Engine::addRequest(const std::string& input, const SamplingParams& sampling_params) {
  auto conversation = nlohmann::json::array({{{"role", "user"}, {"content", input}}});
  auto token_ids = tokenizer_.encode(tokenizer_.applyChatTemplate(conversation));
  addRequest(token_ids, sampling_params);
}

void Engine::addRequest(const std::vector<int32_t>& token_ids,
                        const SamplingParams& sampling_params) {
  auto seq = Sequence::create(Clock::now(), token_ids, cfg_.kvcache_block_size, sampling_params);
  scheduler_.add(seq);
}

void Engine::updateSeqsPerfStats(const std::vector<Sequence::Ptr>& seqs) {
  auto now = Clock::now();
  for (const auto& seq : seqs) {
    if (seq->first_token_ts == Sequence::TimePoint{} && seq->num_tokens > seq->num_prompt_tokens) {
      seq->first_token_ts = now;
    }
    if (seq->state == SequenceState::kFinished) {
      seq->finish_ts = now;
    }
  }
}

void Engine::updateSchedulePerfStats(int num_tokens,
                                     std::chrono::microseconds latency,
                                     bool is_prefill) {
  if (is_prefill) {
    stats_.total_prefill_tokens += num_tokens;
    stats_.total_prefill_latency += latency;
  } else {
    stats_.total_decode_tokens += num_tokens;
    stats_.total_decode_latency += latency;
  }
}

void Engine::logSeqPerfStats(const Sequence::Ptr& seq) {
  auto output_token_count = seq->num_tokens - seq->num_prompt_tokens;
  auto e2e_latency_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(seq->finish_ts - seq->req_ts);
  auto ttft_us =
      std::chrono::duration_cast<std::chrono::microseconds>(seq->first_token_ts - seq->req_ts)
          .count();
  auto decode_latency_us =
      std::chrono::duration_cast<std::chrono::microseconds>(seq->finish_ts - seq->first_token_ts);
  auto tpot_us =
      output_token_count == 0 ? NAN : decode_latency_us.count() / (output_token_count - 1);

  LOG(INFO) << "seq_id=" << seq->seq_id << " prompt_tokens=" << seq->num_prompt_tokens
            << " output_tokens=" << output_token_count << " TTFT=" << (float)ttft_us / 1000.0
            << " ms"
            << " TPOT=" << (float)tpot_us / 1000.0 << " ms"
            << " e2e_latency=" << e2e_latency_ms.count() << " ms";
}

void Engine::logSchedulePerfStats() {
  auto avg_prefill_latency =
      stats_.total_prefill_tokens > 0
          ? (float)stats_.total_prefill_latency.count() / stats_.total_prefill_tokens
          : 0.0f;
  auto avg_decode_latency =
      stats_.total_decode_tokens > 0
          ? (float)stats_.total_decode_latency.count() / stats_.total_decode_tokens
          : 0.0f;

  LOG(INFO) << "Total prefill tokens: " << stats_.total_prefill_tokens << ", prefill throughput: "
            << (float)stats_.total_prefill_tokens /
                   (stats_.total_prefill_latency.count() / 1000000.0)
            << " tok/s";
  LOG(INFO) << "Total decode tokens: " << stats_.total_decode_tokens << ", decode throughput: "
            << (float)stats_.total_decode_tokens / (stats_.total_decode_latency.count() / 1000000.0)
            << " tok/s";
  stats_.reset();
}

std::vector<Sequence::Ptr> Engine::step() {
  auto start = Clock::now();

  // schedule
  auto [seqs, is_prefill] = scheduler_.schedule();
  if (seqs.empty()) return {};

  // run
  auto result = model_runner_.run(seqs, is_prefill);
  if (!result.ok()) {
    LOG(ERROR) << "Inference error: " << result.err();
    return {};
  }

  // postprocess
  auto token_ids = result.value();
  int num_tokens = is_prefill
                       ? std::transform_reduce(seqs.begin(), seqs.end(), 0, std::plus<>(),
                                               [](const auto& seq) { return seq->num_tokens; })
                       : token_ids.size();
  scheduler_.postprocess(seqs, token_ids);
  updateSeqsPerfStats(seqs);

  std::erase_if(seqs,
                [](const Sequence::Ptr& seq) { return seq->state != SequenceState::kFinished; });

  updateSchedulePerfStats(
      num_tokens, std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - start),
      is_prefill);

  return seqs;
}

std::vector<std::string> Engine::generate(
    const std::variant<std::vector<std::string>, std::vector<std::vector<int32_t>>>& prompts,
    const SamplingParams& sampling_params) {
  MATCH(prompts, [this, &sampling_params](const auto& prompt_vector) {
    std::for_each(
        prompt_vector.begin(), prompt_vector.end(),
        [this, &sampling_params](const auto& prompt) { addRequest(prompt, sampling_params); });
  });
  std::map<int, std::string> outputs_m;
  while (!scheduler_.is_finished()) {
    auto seqs = step();
    for (const auto& seq : seqs) {
      logSeqPerfStats(seq);
      outputs_m[seq->seq_id] = tokenizer_.decode(seq->token_ids, true);
    }
  }

  logSchedulePerfStats();

  std::vector<std::string> outputs;
  outputs.reserve(outputs_m.size());
  for (auto& [_, output] : outputs_m) {
    outputs.push_back(std::move(output));
  }
  return outputs;
}

}  // namespace ginfer::engine