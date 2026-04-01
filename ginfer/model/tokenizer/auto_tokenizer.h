#pragma once
#include <tokenizers_cpp.h>
#include <jinja.hpp>
#include <memory>
#include <tuple>
#include <unordered_set>

namespace ginfer::model::tokenizer {

using json = nlohmann::json;

class AutoTokenizer {
 public:
  AutoTokenizer(const std::string& model_path);
  std::vector<int32_t> encode(const std::string& text);
  std::tuple<std::vector<int32_t>, std::vector<int>> encodeBatch(
      const std::vector<std::string>& texts);
  std::string decode(const std::vector<int32_t>& token_ids, bool skip_special_tokens = false);
  std::vector<std::string> decodeBatch(const std::vector<int32_t>& token_ids,
                                       const std::vector<int>& cu_seqlens,
                                       bool skip_special_tokens = false);
  std::string applyChatTemplate(const json& conversation,
                                bool add_generation_prompt = true,
                                const json& tools = json::array());

 private:
  void constructChatTemplate();
  void constructTokenizer();
  std::string decodeImpl(std::vector<int32_t>& token_ids, bool skip_special_tokens);

 private:
  std::string model_path_;
  std::string bos_token_;
  std::unordered_set<int32_t> special_token_ids_;
  std::unique_ptr<tokenizers::Tokenizer> tokenizer_;
  std::unique_ptr<jinja::Template> chat_template_;
};

}  // namespace ginfer::model::tokenizer
