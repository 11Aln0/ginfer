#include <gtest/gtest.h>
#include <cstdlib>

#include "ginfer/common/device.h"
#include "ginfer/engine/config.h"
#include "ginfer/engine/engine.h"
#include "ginfer/model/model_factory.h"

namespace ginfer::test {

namespace {

engine::Config makeEngineConfig(const std::string& model_path) {
  auto loader = model::ModelFactory::createLoader(model_path);
  const auto& model_cfg = loader->getModelConfig();

  return {
      .model_path = model_path,
      .device_type = common::DeviceType::kDeviceCUDA,
      .max_num_batched_tokens = 512,
      .max_num_seqs = 4,
      .max_seq_len = 512,
      .gpu_memory_utilization = 0.8f,
      .kvcache_block_size = 16,
      .model_config = model_cfg,
  };
}

}  // namespace

TEST(EngineTest, GenerateReturnsOneOutputPerPrompt) {
  const char* model_path = std::getenv("MODEL_PATH");
  ASSERT_NE(model_path, nullptr) << "MODEL_PATH environment variable not set";

  engine::Engine engine(makeEngineConfig(model_path));

  const std::vector<std::string> prompts = {"Who are you?"};
  auto start = std::chrono::high_resolution_clock::now();
  auto outputs = engine.generate(prompts);
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::high_resolution_clock::now() - start);
  LOG(INFO) << "ginfer generate time: " << duration.count() << " ms";

  ASSERT_EQ(outputs.size(), prompts.size());
  for (const auto& output : outputs) {
    EXPECT_FALSE(output.empty());
    LOG(INFO) << "Generated output: " << output;
  }
}

}  // namespace ginfer::test
