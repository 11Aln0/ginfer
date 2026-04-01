#include "ginfer/model/tokenizer/auto_tokenizer.h"
#include <algorithm>
#include <iterator>
#include <utility>
#include "ginfer/common/errors.h"
#include "ginfer/utils/utils.h"

namespace ginfer::model::tokenizer {

AutoTokenizer::AutoTokenizer(const std::string& model_path) : model_path_(model_path) {
  constructChatTemplate();
  constructTokenizer();
}

void AutoTokenizer::constructChatTemplate() {
  std::string tok_cfg_path = model_path_ + "/tokenizer_config.json";
  auto tok_cfg_blob = utils::file::loadBytesFromFile(tok_cfg_path);
  CHECK_THROW(tok_cfg_blob.ok(), "Failed to load tokenizer config file: {}", tok_cfg_blob.err());
  auto tok_cfg_json = nlohmann::json::parse(tok_cfg_blob.value());
  std::string chat_template_str = tok_cfg_json["chat_template"];
  chat_template_ = std::make_unique<jinja::Template>(chat_template_str);
}

void AutoTokenizer::constructTokenizer() {
  auto tok_blob = utils::file::loadBytesFromFile(model_path_ + "/tokenizer.json");
  CHECK_THROW(tok_blob.ok(), "Failed to load tokenizer file: {}", tok_blob.err());
  auto tok_json = nlohmann::json::parse(tok_blob.value());

  for (const auto& token : tok_json["added_tokens"]) {
    if (token.value("special", false)) {
      special_token_ids_.insert(token["id"].get<int32_t>());
    }
  }

  tokenizer_ = tokenizers::Tokenizer::FromBlobJSON(tok_blob.value());
}

std::vector<int32_t> AutoTokenizer::encode(const std::string& text) {
  return tokenizer_->Encode(text);
}

std::tuple<std::vector<int32_t>, std::vector<int>> AutoTokenizer::encodeBatch(
    const std::vector<std::string>& texts) {
  auto encoded = tokenizer_->EncodeBatch(texts);
  std::vector<int> cu_seqlens;
  cu_seqlens.reserve(encoded.size() + 1);
  cu_seqlens.push_back(0);

  size_t total_tokens = 0;
  for (const auto& ids : encoded) {
    total_tokens += ids.size();
  }

  std::vector<int32_t> flat_token_ids;
  flat_token_ids.reserve(total_tokens);

  int prefix_sum = 0;
  for (const auto& ids : encoded) {
    flat_token_ids.insert(flat_token_ids.end(), ids.begin(), ids.end());
    prefix_sum += static_cast<int>(ids.size());
    cu_seqlens.push_back(prefix_sum);
  }
  return {std::move(flat_token_ids), std::move(cu_seqlens)};
}

std::string AutoTokenizer::decodeImpl(std::vector<int32_t>& token_ids, bool skip_special_tokens) {
  if (!skip_special_tokens) return tokenizer_->Decode(token_ids);

  size_t write_idx = 0;
  for (size_t read_idx = 0; read_idx < token_ids.size(); ++read_idx) {
    if (special_token_ids_.count(token_ids[read_idx])) continue;
    token_ids[write_idx++] = token_ids[read_idx];
  }
  token_ids.resize(write_idx);
  return tokenizer_->Decode(token_ids);
}

std::string AutoTokenizer::decode(const std::vector<int32_t>& token_ids, bool skip_special_tokens) {
  if (!skip_special_tokens) return tokenizer_->Decode(token_ids);
  auto token_ids_copy = token_ids;
  return decodeImpl(token_ids_copy, skip_special_tokens);
}

std::vector<std::string> AutoTokenizer::decodeBatch(const std::vector<int32_t>& token_ids,
                                                    const std::vector<int>& cu_seqlens,
                                                    bool skip_special_tokens) {
  std::vector<std::string> texts;
  if (cu_seqlens.size() <= 1) return texts;
  texts.reserve(cu_seqlens.size() - 1);

  int max_seq_len = 0;
  for (size_t i = 0; i + 1 < cu_seqlens.size(); ++i) {
    max_seq_len = std::max(max_seq_len, cu_seqlens[i + 1] - cu_seqlens[i]);
  }

  std::vector<int32_t> seq_token_ids;
  seq_token_ids.reserve(max_seq_len);
  for (size_t i = 0; i + 1 < cu_seqlens.size(); ++i) {
    seq_token_ids.assign(token_ids.begin() + cu_seqlens[i], token_ids.begin() + cu_seqlens[i + 1]);
    texts.push_back(decodeImpl(seq_token_ids, skip_special_tokens));
  }
  return texts;
}

std::string AutoTokenizer::applyChatTemplate(const json& conversation,
                                             bool add_generation_prompt,
                                             const json& tools) {
  return chat_template_->apply_chat_template(conversation, add_generation_prompt, tools);
}

}  // namespace ginfer::model::tokenizer
