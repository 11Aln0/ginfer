#include "ginfer/model/tokenizer/auto_tokenizer.h"
#include "ginfer/common/errors.h"
#include "ginfer/utils/utils.h"
#include <algorithm>
#include <iterator>

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

std::vector<int32_t> AutoTokenizer::encode(const std::string& text) { return tokenizer_->Encode(text); }

std::string AutoTokenizer::decode(const std::vector<int32_t>& token_ids, bool skip_special_tokens) {
  if (!skip_special_tokens) return tokenizer_->Decode(token_ids);
  std::vector<int32_t> filtered_ids;
  std::copy_if(token_ids.begin(), token_ids.end(), std::back_inserter(filtered_ids),
               [this](int32_t id) { return !special_token_ids_.count(id); });
  return tokenizer_->Decode(filtered_ids);
}

std::string AutoTokenizer::applyChatTemplate(const json& conversation, bool add_generation_prompt, const json& tools) {
  return chat_template_->apply_chat_template(conversation, add_generation_prompt, tools);
}

}  // namespace ginfer::model::tokenizer
