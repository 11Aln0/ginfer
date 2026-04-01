#include "ginfer/engine/engine.h"
#include <map>

namespace ginfer::engine {

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
  scheduler_.add(seq);
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