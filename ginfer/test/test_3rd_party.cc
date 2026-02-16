#include <glog/logging.h>
#include <gtest/gtest.h>
#include <tokenizers_cpp.h>
#include <fstream>
#include <iostream>
#include <jinja.hpp>
#include <nlohmann/json.hpp>
#include "ginfer/utils/utils.h"

namespace ginfer::test {

TEST(LogTest, Basic) {
  LOG(INFO) << "This is an info log.";
  LOG(WARNING) << "This is a warning log.";
  LOG(ERROR) << "This is an error log.";
}

TEST(JSONTest, Parse) {
  std::string json_str = R"({"name": "John", "age": 30, "city": "New York"})";
  auto json_obj = nlohmann::json::parse(json_str);

  EXPECT_EQ(json_obj["name"], "John");
  EXPECT_EQ(json_obj["age"], 30);
  EXPECT_EQ(json_obj["city"], "New York");
}

void testTokenizer(std::unique_ptr<tokenizers::Tokenizer> tok, bool print_vocab = false) {
  // Check #1. Encode and Decode
  std::string prompt = "What is the  capital of Canada?";
  std::vector<int> ids = tok->Encode(prompt);
  std::string decoded_prompt = tok->Decode(ids);
  EXPECT_EQ(prompt, decoded_prompt) << "prompt=\"" << prompt << "\", decoded_prompt=\"" << decoded_prompt << "\"";

  // Check #2. IdToToken and TokenToId
  std::vector<int32_t> ids_to_test = {0, 1, 2, 3, 32, 33, 34, 130, 131, 1000};
  for (auto id : ids_to_test) {
    auto token = tok->IdToToken(id);
    auto id_new = tok->TokenToId(token);
    EXPECT_EQ(id, id_new) << "id=" << id << ", token=\"" << token << "\", id_new=" << id_new;
  }

  // Check #3. GetVocabSize
  auto vocab_size = tok->GetVocabSize();
  std::cout << "vocab_size=" << vocab_size << std::endl;

  std::cout << std::endl;
}

TEST(TokenizersTest, Basic) {
  auto start = std::chrono::high_resolution_clock::now();

  // Read blob from file.
  const char* tokenizer_path = std::getenv("TOKENIZER_PATH");
  EXPECT_NE(tokenizer_path, nullptr) << "TOKENIZER_PATH environment variable not set";
  auto blob = utils::file::loadBytesFromFile(tokenizer_path);
  EXPECT_TRUE(blob.ok()) << "Failed to load tokenizer file: " << blob.err();
  // Note: all the current factory APIs takes in-memory blob as input.
  // This gives some flexibility on how these blobs can be read.
  auto tok = tokenizers::Tokenizer::FromBlobJSON(blob.value());

  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

  std::cout << "Load time: " << duration << " ms" << std::endl;

  testTokenizer(std::move(tok), false);
}

TEST(JinjaTest, Basic) {
  // Read blob from file.
  const char* tokenizer_config_path = std::getenv("TOKENIZER_CONFIG_PATH");
  EXPECT_NE(tokenizer_config_path, nullptr) << "TOKENIZER_CONFIG_PATH environment variable not set";
  auto blob = utils::file::loadBytesFromFile(tokenizer_config_path);
  EXPECT_TRUE(blob.ok()) << "Failed to load tokenizer config file: " << blob.err();

  auto tok_cfg_json = nlohmann::json::parse(blob.value());

  auto chat_template_str = tok_cfg_json["chat_template"].get<std::string>();
  jinja::Template chat_tpl = jinja::Template(chat_template_str);

  jinja::json messages = nlohmann::json::array({{{"role", "user"}, {"content", "Who are you?"}}});

  auto prompt = chat_tpl.apply_chat_template(messages, true, nlohmann::json::array());

  std::cout << "Generated prompt:\n" << prompt << std::endl;
}

}  // namespace ginfer::test