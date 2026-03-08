#pragma once
#include <tokenizers_cpp.h>
#include <jinja.hpp>
#include <memory>
#include <unordered_set>

namespace ginfer::core::model::tokenizer {

using json = nlohmann::json;

class AutoTokenizer {
 public:
  AutoTokenizer(const std::string& model_path);
  std::vector<int32_t> encode(const std::string& text);
  std::string decode(const std::vector<int32_t>& token_ids, bool skip_special_tokens = false);
  std::string applyChatTemplate(const json& conversation,
                                bool add_generation_prompt = true,
                                const json& tools = json::array());

 private:
  void constructChatTemplate();
  void constructTokenizer();

 private:
  std::string model_path_;
  std::unordered_set<int32_t> special_token_ids_;
  std::unique_ptr<tokenizers::Tokenizer> tokenizer_;
  std::unique_ptr<jinja::Template> chat_template_;
};

}  // namespace ginfer::core::model::tokenizer
