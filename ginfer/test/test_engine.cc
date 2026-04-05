#include <gtest/gtest.h>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include "ginfer/common/device.h"
#include "ginfer/engine/config.h"
#include "ginfer/engine/engine.h"
#include "ginfer/model/model_factory.h"

namespace ginfer::test {

TEST(EngineTest, Generate) {
  const char* model_path = std::getenv("MODEL_PATH");
  ASSERT_NE(model_path, nullptr) << "MODEL_PATH environment variable not set";
  auto loader = model::ModelFactory::createLoader(model_path);
  const auto& model_cfg = loader->getModelConfig();

  engine::Config config = {
      .model_path = model_path,
      .device_type = common::DeviceType::kDeviceCUDA,
      .max_num_batched_tokens = 2048,
      .max_num_seqs = 40,
      .max_seq_len = 512,
      .gpu_memory_utilization = 0.7f,
      .kvcache_block_size = 16,
      .model_config = model_cfg,
  };
  engine::Engine eng(config);

  const std::vector<std::string> prompts = {
      "Explain the difference between supervised and unsupervised learning.",
      "Write a short story about a robot who discovers it can feel emotions.",
      "What are the main causes of the French Revolution?",
      "Summarize the key principles of object-oriented programming.",
      "Describe the process of photosynthesis in simple terms.",
      "What would happen if humans could photosynthesize like plants?",
      "Compare and contrast TCP and UDP protocols.",
      "Write a Python function that checks if a string is a palindrome.",
      "What are the ethical implications of artificial general intelligence?",
      "Explain how black holes form and what happens at the event horizon.",
  };
  auto start = std::chrono::high_resolution_clock::now();
  auto outputs = eng.generate(prompts, engine::SamplingParams{.max_tokens = 1024});
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::high_resolution_clock::now() - start);
  LOG(INFO) << "ginfer generate time: " << duration.count() << " ms";

  ASSERT_EQ(outputs.size(), prompts.size());
  for (const auto& output : outputs) {
    EXPECT_FALSE(output.empty());
    LOG(INFO) << "Generated output: " << output;
  }
}

TEST(EngineTest, Benchmark) {
  const char* model_path = std::getenv("MODEL_PATH");
  ASSERT_NE(model_path, nullptr) << "MODEL_PATH environment variable not set";
  auto loader = model::ModelFactory::createLoader(model_path);
  const auto& model_cfg = loader->getModelConfig();

  constexpr int numSeqs = 256;
  constexpr int inputLen = 512;
  constexpr int outputLen = 128;

  engine::Config config = {
      .model_path = model_path,
      .device_type = common::DeviceType::kDeviceCUDA,
      .max_num_batched_tokens = 16384,
      .max_num_seqs = numSeqs,
      .max_seq_len = 16384,
      .gpu_memory_utilization = 0.8f,
      .kvcache_block_size = 16,
      .model_config = model_cfg,
  };
  engine::Engine eng(config);

  std::mt19937 rng(0);
  // std::uniform_int_distribution<int> prompt_len_dist(100, kMaxInputLen);
  std::uniform_int_distribution<int32_t> token_id_dist(0, 10000);

  std::vector<std::vector<int32_t>> prompt_token_ids;
  prompt_token_ids.reserve(numSeqs);
  for (int i = 0; i < numSeqs; ++i) {
    int prompt_len = inputLen;
    auto& token_ids = prompt_token_ids.emplace_back();
    token_ids.reserve(prompt_len);
    for (int j = 0; j < prompt_len; ++j) {
      token_ids.push_back(token_id_dist(rng));
    }
  }

  eng.generate({std::vector<std::string>{"Benchmark warmup."}},
               engine::SamplingParams{.max_tokens = 64});

  auto start = std::chrono::steady_clock::now();
  auto outputs = eng.generate(prompt_token_ids,
                              engine::SamplingParams{.max_tokens = outputLen, .ignore_eos = true});
  auto duration = std::chrono::duration_cast<std::chrono::duration<double>>(
      std::chrono::steady_clock::now() - start);

  ASSERT_EQ(outputs.size(), prompt_token_ids.size());
  for (const auto& output : outputs) {
    EXPECT_FALSE(output.empty());
  }

  auto total_target_tokens = static_cast<int64_t>(numSeqs) * (inputLen + outputLen);
  auto throughput = total_target_tokens / duration.count();
  LOG(INFO) << "Benchmark total target tokens: " << total_target_tokens
            << ", time: " << duration.count() << " s, throughput: " << throughput << " tok/s";
}

}  // namespace ginfer::test
