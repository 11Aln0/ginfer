#include "ginfer/engine/model_runner.h"
#include <ranges>
#include <vector>
#include "ginfer/core/memory/allocator_factory.h"
#include "ginfer/core/model/model_factory.h"
#include "ginfer/core/tensor/dtype.h"
#include "ginfer/core/tensor/tensor.h"

namespace ginfer::engine {

ModelRunner::ModelRunner(Config config) : config_(std::move(config)) {
  allocator_ = core::memory::getDefaultDeviceAllocator(config_.device_type);
  pooled_allocator_ =
      core::memory::getDeviceAllocator<core::memory::PooledAllocStrategy>(config_.device_type);
  loadModel();
  warmupModel();
  allocateKVCache();
}

void ModelRunner::loadModel() {
  const auto& cfg = config_;
  auto loader = core::model::ModelFactory::createLoader(cfg.model_path);
  model_ = loader->load();
  model_->toDevice(cfg.device_type);
  model_->setMaxSeqLen(cfg.max_seq_len);
}

void ModelRunner::warmupModel() {
  auto dev_type = model_->getDeviceType();
  auto dtype = model_->getDtype();
  DECLARE_OR_THROW(input_ids, Tensor::create(dtype, {1}, DeviceType::kDeviceCPU));
  THROW_ON_ERR(input_ids->toDevice(dev_type));
  DECLARE_OR_THROW(positions, Tensor::create(dtype, {1}, DeviceType::kDeviceCPU));
  THROW_ON_ERR(positions->toDevice(dev_type));
  auto ctx = core::InferContext().setDeviceContext(common::DeviceContext::create(dev_type));
  THROW_ON_ERR(model_->predict(ctx, input_ids, positions));
}

void ModelRunner::allocateKVCache() {
  auto dev_type = model_->getDeviceType();
  auto mem_info = core::memory::getDefaultDeviceAllocator(dev_type)->getMemInfo();
  auto [num_heads, num_kv_heads, head_dim] = model_->getAttentionConfig();
  auto dtype = model_->getDtype();
  auto n_layer = model_->getNumLayers();

  size_t block_bytes = 2 * num_kv_heads * head_dim * core::tensor::dTypeSize(dtype) * n_layer;
  int64_t num_kvcache_blocks = mem_info.free / block_bytes;
  this->num_kvcache_blocks_ = num_kvcache_blocks;

  CHECK(num_kvcache_blocks > 0) << "Not enough memory for KV cache";

  DECLARE_OR_THROW(
      kv_cache,
      Tensor::create(dtype, {2, n_layer, num_kvcache_blocks, num_kv_heads, head_dim}, dev_type));
  auto k_cache =
      kv_cache->slice(0, 0, 1)->reshape({n_layer, num_kvcache_blocks, num_kv_heads, head_dim});
  auto v_cache =
      kv_cache->slice(0, 1, 2)->reshape({n_layer, num_kvcache_blocks, num_kv_heads, head_dim});

  auto layer_kv_shape = core::tensor::Shape{num_kvcache_blocks, num_kv_heads, head_dim};
  for (int i = 0; i < n_layer; ++i) {
    auto layer_k_cache = k_cache->slice(0, i, i + 1)->reshape(layer_kv_shape);
    auto layer_v_cache = v_cache->slice(0, i, i + 1)->reshape(layer_kv_shape);
    model_->setKVCache(i, layer_k_cache, layer_v_cache);
  }
}

void ModelRunner::prepareBlockTables(std::vector<Sequence::Ptr>& seqs) {
  int max_block_tbl_len =
      std::ranges::max(seqs, {}, [](const auto& seq) { return seq->numBlocks(); })->numBlocks();
}

void ModelRunner::preparePrefill(std::vector<Sequence::Ptr>& seqs) {
  std::vector<int32_t> input_ids;
  std::vector<int32_t> positions;
  std::vector<int32_t> cu_seqlen_q = {0};
  std::vector<int32_t> cu_seqlen_kv = {0};
  std::vector<int32_t> slot_mapping;
  int max_seqlen_q = 0;

  for (auto& seq : seqs) {
    int seqlen_q = seq->num_tokens - seq->num_cached_tokens;
    int n_blocks = seq->numBlocks();
    int n_tokens = seq->num_tokens;

    input_ids.insert(input_ids.end(), seq->token_ids.begin() + seq->num_cached_tokens,
                     seq->token_ids.end());
    for (int i = seq->num_cached_tokens; i < n_tokens; ++i) {
      positions.push_back(i);
    }

    cu_seqlen_q.push_back(cu_seqlen_q.back() + seqlen_q);
    cu_seqlen_kv.push_back(cu_seqlen_kv.back() + seq->num_tokens);
    max_seqlen_q = std::max(max_seqlen_q, seqlen_q);

    for (int i = seq->numCachedBlocks(); i < n_blocks; ++i) {
      int block_id = seq->block_table[i];
      int start = block_id * seq->block_size;
      int end = i == n_blocks - 1 ? (n_tokens - start) : (block_id + 1) * seq->block_size;
      auto slot_range = std::views::iota(start, end);
      slot_mapping.insert(slot_mapping.end(), slot_range.begin(), slot_range.end());
    }
  }

  if (cu_seqlen_q.back() < cu_seqlen_kv.back()) {
  }
}

}  // namespace ginfer::engine