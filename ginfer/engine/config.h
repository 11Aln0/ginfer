#pragma once
#include <string>
#include "ginfer/common/device.h"
#include "ginfer/model/model.h"

namespace ginfer::engine {

struct Config {
  std::string model_path;

  common::DeviceType device_type;

  int max_num_batched_tokens;
  int max_num_seqs;
  int max_seq_len;

  float gpu_memory_utilization;
  int kvcache_block_size;

  model::ModelConfig model_config;
};

}  // namespace ginfer::engine