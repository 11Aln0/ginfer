#include "ginfer/engine/model_runner.h"
#include <glog/logging.h>
#include <algorithm>
#include <ranges>
#include <tuple>
#include <vector>
#include "ginfer/core/memory/allocator_factory.h"
#include "ginfer/core/tensor/dtype.h"
#include "ginfer/core/tensor/tensor.h"
#include "ginfer/core/tensor/tensor_writer.h"
#include "ginfer/model/model_factory.h"

namespace ginfer::engine {

ModelRunner::ModelRunner(const Config& config) : config_(config) {
  ctx = core::InferContext().setDeviceContext(common::DeviceContext::create(config.device_type));
  loadModel();
  warmupModel();
  allocateWorkspace();
  allocateKVCache();
}

void ModelRunner::loadModel() {
  const auto& cfg = config_;
  auto loader = model::ModelFactory::createLoader(cfg.model_path);
  model_ = loader->load();
  model_->setRuntimeConfig({
      .max_seq_len = cfg.max_seq_len,
      .max_batch_size = cfg.max_num_seqs,
  });
  model_->toDevice(cfg.device_type);
}

void ModelRunner::warmupModel() {
  using core::tensor::DataType;

  auto dev_type = model_->getDeviceType();
  auto dtype = model_->getConfig().dtype;
  DECLARE_OR_THROW(input_ids,
                   Tensor::create(DataType::kDataTypeInt32, {1}, DeviceType::kDeviceCPU));
  input_ids->data<int32_t>()[0] = 0;  // dummy token id
  ASSIGN_OR_THROW(input_ids, input_ids->toDevice(dev_type));
  DECLARE_OR_THROW(positions,
                   Tensor::create(DataType::kDataTypeInt32, {1}, DeviceType::kDeviceCPU));
  positions->data<int32_t>()[0] = 0;  // dummy position
  ASSIGN_OR_THROW(positions, positions->toDevice(dev_type));
  auto ctx = core::InferContext().setDeviceContext(common::DeviceContext::create(dev_type));
  THROW_ON_ERR(model_->predict(ctx, input_ids, positions));
  LOG(INFO) << "Model warmup completed.";
}

void ModelRunner::allocateKVCache() {
  auto dev_type = model_->getDeviceType();
  auto mem_info = core::memory::getDefaultDeviceAllocator(dev_type)->getMemInfo();

  auto& cfg = model_->getConfig();
  int num_kv_heads = cfg.num_kv_heads, head_dim = cfg.head_dim;
  auto n_layer = cfg.nlayer;
  int block_size = config_.kvcache_block_size;

  size_t block_bytes =
      2 * num_kv_heads * head_dim * core::tensor::dTypeSize(cfg.dtype) * n_layer * block_size;
  int64_t num_kvcache_blocks = static_cast<int64_t>(static_cast<float>(mem_info.free) *
                                                    config_.gpu_memory_utilization / block_bytes);
  LOG(INFO) << "Estimated number of KV cache blocks that can fit in memory: " << num_kvcache_blocks
            << " (free memory: " << mem_info.free / (1024.0 * 1024.0)
            << " MB, block size in bytes: " << block_bytes << " bytes)";
  this->num_kvcache_blocks_ = num_kvcache_blocks;

  CHECK(num_kvcache_blocks > 0) << "Not enough memory for KV cache";

  DECLARE_OR_THROW(
      kv_cache, Tensor::create(cfg.dtype,
                               {2, n_layer, num_kvcache_blocks, block_size, num_kv_heads, head_dim},
                               dev_type));
  auto k_cache = kv_cache->slice(0, 0, 1)->reshape(
      {n_layer, num_kvcache_blocks, block_size, num_kv_heads, head_dim});
  auto v_cache = kv_cache->slice(0, 1, 2)->reshape(
      {n_layer, num_kvcache_blocks, block_size, num_kv_heads, head_dim});

  auto layer_kv_shape = core::tensor::Shape{num_kvcache_blocks, block_size, num_kv_heads, head_dim};
  for (int i = 0; i < n_layer; ++i) {
    auto layer_k_cache = k_cache->slice(0, i, i + 1)->reshape(layer_kv_shape);
    auto layer_v_cache = v_cache->slice(0, i, i + 1)->reshape(layer_kv_shape);
    model_->setKVCache(i, layer_k_cache, layer_v_cache);
  }
}

void ModelRunner::resetContext(core::InferContext& ctx) {
  ctx.max_seqlen_q.reset();
  ctx.cu_seqlens_q.reset();
  ctx.cu_seqlens_kv.reset();
  ctx.block_tables.reset();
  ctx.slot_mapping.reset();
  ctx.is_prefill = false;
}

int ModelRunner::getMaxBlocksPerSeq() const {
  return (config_.max_num_batched_tokens + config_.kvcache_block_size - 1) /
         config_.kvcache_block_size;
}

void ModelRunner::allocateWorkspace() {
  int max_blocks_per_seq = getMaxBlocksPerSeq();

  auto max_num_seqs = config_.max_num_seqs;
  auto max_num_batched_tokens = config_.max_num_batched_tokens;

  auto host_allocator =
      core::memory::getDeviceAllocator(DeviceType::kDeviceCPU, core::memory::kPooled);
  auto dev_allocator = core::memory::getDefaultDeviceAllocator(config_.device_type);

  auto& host = host_workspace_;
  auto& dev = dev_workspace_;

  ASSIGN_OR_THROW(host.input_ids, Tensor::create(core::tensor::DataType::kDataTypeInt32,
                                                 {max_num_batched_tokens}, host_allocator));
  ASSIGN_OR_THROW(host.positions, Tensor::create(core::tensor::DataType::kDataTypeInt32,
                                                 {max_num_batched_tokens}, host_allocator));
  ASSIGN_OR_THROW(host.cu_seqlens_q, Tensor::create(core::tensor::DataType::kDataTypeInt32,
                                                    {max_num_seqs + 1}, host_allocator));
  ASSIGN_OR_THROW(host.cu_seqlens_kv, Tensor::create(core::tensor::DataType::kDataTypeInt32,
                                                     {max_num_seqs + 1}, host_allocator));
  ASSIGN_OR_THROW(host.slot_mapping, Tensor::create(core::tensor::DataType::kDataTypeInt32,
                                                    {max_num_batched_tokens}, host_allocator));
  ASSIGN_OR_THROW(host.block_tables,
                  Tensor::create(core::tensor::DataType::kDataTypeInt32,
                                 {max_num_seqs, max_blocks_per_seq}, host_allocator));

  ASSIGN_OR_THROW(dev.input_ids, Tensor::create(core::tensor::DataType::kDataTypeInt32,
                                                {max_num_batched_tokens}, dev_allocator));
  ASSIGN_OR_THROW(dev.positions, Tensor::create(core::tensor::DataType::kDataTypeInt32,
                                                {max_num_batched_tokens}, dev_allocator));
  ASSIGN_OR_THROW(dev.cu_seqlens_q, Tensor::create(core::tensor::DataType::kDataTypeInt32,
                                                   {max_num_seqs + 1}, dev_allocator));
  ASSIGN_OR_THROW(dev.cu_seqlens_kv, Tensor::create(core::tensor::DataType::kDataTypeInt32,
                                                    {max_num_seqs + 1}, dev_allocator));
  ASSIGN_OR_THROW(dev.slot_mapping, Tensor::create(core::tensor::DataType::kDataTypeInt32,
                                                   {max_num_batched_tokens}, dev_allocator));
  ASSIGN_OR_THROW(dev.block_tables,
                  Tensor::create(core::tensor::DataType::kDataTypeInt32,
                                 {max_num_seqs, max_blocks_per_seq}, dev_allocator));
}

ModelRunner::Workspace ModelRunner::prepareWorkspace(ModelRunner::Workspace& from,
                                                     int total_q_tokens,
                                                     int batch_size) {
  CHECK(total_q_tokens <= config_.max_num_batched_tokens)
      << "total_q_tokens exceeds workspace capacity";
  CHECK(batch_size <= config_.max_num_seqs) << "batch_size exceeds workspace capacity";

  auto input_ids = from.input_ids->slice(0, 0, total_q_tokens);
  auto positions = from.positions->slice(0, 0, total_q_tokens);
  auto cu_seqlens_q = from.cu_seqlens_q->slice(0, 0, batch_size + 1);
  auto cu_seqlens_kv = from.cu_seqlens_kv->slice(0, 0, batch_size + 1);
  auto slot_mapping = from.slot_mapping->slice(0, 0, total_q_tokens);
  auto block_tables = from.block_tables->slice(0, 0, batch_size);

  return {input_ids, positions, cu_seqlens_q, cu_seqlens_kv, slot_mapping, block_tables};
}

void ModelRunner::prepareBlockTables(std::vector<Sequence::Ptr>& seqs,
                                     TensorRef& block_tables_host) {
  CHECK(!seqs.empty()) << "prepareBlockTables requires non-empty sequences";
  int max_block_tbl_len = std::ranges::max(seqs, {}, [](const auto& seq) {
                            return seq->block_table.size();
                          })->block_table.size();
  CHECK_LE(max_block_tbl_len, getMaxBlocksPerSeq())
      << "block table length exceeds workspace capacity";

  auto block_tables = core::tensor::bindTensor2D<int32_t>(block_tables_host);
  for (const auto& seq : seqs) {
    block_tables.appendRow(seq->block_table.begin(), seq->block_table.end(), max_block_tbl_len, -1);
  }
}

std::tuple<core::tensor::TensorRef, core::tensor::TensorRef> ModelRunner::prepareDecode(
    core::InferContext& ctx, std::vector<Sequence::Ptr>& seqs) {
  CHECK_LE(seqs.size(), static_cast<size_t>(config_.max_num_seqs)) << "too many sequences";

  int total_q_tokens = seqs.size();
  CHECK_LE(total_q_tokens, config_.max_num_batched_tokens)
      << "batched token count exceeds workspace capacity";

  auto host = prepareWorkspace(host_workspace_, total_q_tokens, seqs.size());

  auto input_ids = core::tensor::bindTensor<int32_t>(host.input_ids);
  auto positions = core::tensor::bindTensor<int32_t>(host.positions);
  auto cu_seqlens_kv = core::tensor::bindTensor<int32_t>(host.cu_seqlens_kv);
  auto slot_mapping = core::tensor::bindTensor<int32_t>(host.slot_mapping);

  cu_seqlens_kv.append(0);

  const int block_size = config_.kvcache_block_size;

  for (auto& seq : seqs) {
    CHECK_GT(seq->num_tokens, 0) << "decode requires non-empty sequences";

    input_ids.append(seq->token_ids.back());
    positions.append(seq->num_tokens - 1);
    cu_seqlens_kv.append(cu_seqlens_kv.back() + seq->num_tokens);

    int token_idx = seq->num_tokens - 1;
    int block_idx = token_idx / block_size;
    int block_offset = token_idx % block_size;
    CHECK_LT(block_idx, seq->block_table.size())
        << "decode token block index exceeds block table size";
    int slot = seq->block_table[block_idx] * block_size + block_offset;
    slot_mapping.append(slot);
  }

  prepareBlockTables(seqs, host.block_tables);

  auto dev = prepareWorkspace(dev_workspace_, total_q_tokens, seqs.size());

  dev.input_ids->copyFrom(host.input_ids, true);
  dev.positions->copyFrom(host.positions, true);
  dev.cu_seqlens_kv->copyFrom(host.cu_seqlens_kv, true);
  dev.slot_mapping->copyFrom(host.slot_mapping, true);
  dev.block_tables->copyFrom(host.block_tables, true);

  ctx.setCuSeqlensKV(dev.cu_seqlens_kv);
  ctx.setSlotMapping(dev.slot_mapping);
  ctx.setBlockTables(dev.block_tables);
  ctx.setIsPrefill(false);

  return {dev.input_ids, dev.positions};
}

std::tuple<core::tensor::TensorRef, core::tensor::TensorRef> ModelRunner::preparePrefill(
    core::InferContext& ctx, std::vector<Sequence::Ptr>& seqs) {
  CHECK_LE(seqs.size(), static_cast<size_t>(config_.max_num_seqs)) << "too many sequences";

  int total_q_tokens = 0;
  int max_seqlen_q = 0;
  for (const auto& seq : seqs) {
    int seqlen_q = seq->num_tokens - seq->num_cached_tokens;
    total_q_tokens += seqlen_q;
    max_seqlen_q = std::max(max_seqlen_q, seqlen_q);
  }

  CHECK_LE(total_q_tokens, config_.max_num_batched_tokens)
      << "batched token count exceeds workspace capacity";

  auto host = prepareWorkspace(host_workspace_, total_q_tokens, seqs.size());
  auto dev = prepareWorkspace(dev_workspace_, total_q_tokens, seqs.size());

  auto input_ids = core::tensor::bindTensor<int32_t>(host.input_ids);
  auto positions = core::tensor::bindTensor<int32_t>(host.positions);
  auto cu_seqlens_q = core::tensor::bindTensor<int32_t>(host.cu_seqlens_q);
  auto cu_seqlens_kv = core::tensor::bindTensor<int32_t>(host.cu_seqlens_kv);
  auto slot_mapping = core::tensor::bindTensor<int32_t>(host.slot_mapping);

  cu_seqlens_q.append(0);
  cu_seqlens_kv.append(0);

  const int block_size = config_.kvcache_block_size;

  for (auto& seq : seqs) {
    int seqlen_q = seq->num_tokens - seq->num_cached_tokens;
    int n_blocks = seq->numBlocks();
    int n_tokens = seq->num_tokens;

    input_ids.extend(seq->token_ids.begin() + seq->num_cached_tokens, seq->token_ids.end());
    auto positions_iota = std::views::iota(seq->num_cached_tokens, n_tokens);
    positions.extend(positions_iota.begin(), positions_iota.end());

    cu_seqlens_q.append(cu_seqlens_q.back() + seqlen_q);
    cu_seqlens_kv.append(cu_seqlens_kv.back() + seq->num_tokens);

    for (int i = seq->numCachedBlocks(); i < n_blocks; ++i) {
      int block_id = seq->block_table[i];
      int start = block_id * block_size;
      int end = i == n_blocks - 1 ? start + (n_tokens - i * block_size) : start + block_size;
      for (int slot = start; slot < end; ++slot) {
        slot_mapping.append(slot);
      }
    }
  }

  prepareBlockTables(seqs, host.block_tables);

  dev.input_ids->copyFrom(host.input_ids, true);
  dev.positions->copyFrom(host.positions, true);
  dev.cu_seqlens_q->copyFrom(host.cu_seqlens_q, true);
  dev.cu_seqlens_kv->copyFrom(host.cu_seqlens_kv, true);
  dev.slot_mapping->copyFrom(host.slot_mapping, true);
  dev.block_tables->copyFrom(host.block_tables, true);

  ctx.setCuSeqlensQ(dev.cu_seqlens_q);
  ctx.setCuSeqlensKV(dev.cu_seqlens_kv);
  ctx.setSlotMapping(dev.slot_mapping);
  ctx.setBlockTables(dev.block_tables);
  ctx.setMaxSeqlenQ(max_seqlen_q);
  ctx.setIsPrefill(true);

  return {dev.input_ids, dev.positions};
}

Result<std::vector<int32_t>, std::string> ModelRunner::run(std::vector<Sequence::Ptr>& seqs,
                                                           bool is_prefill) {
  std::tuple<core::tensor::TensorRef, core::tensor::TensorRef> inputs;
  if (is_prefill) {
    inputs = preparePrefill(ctx, seqs);
  } else {
    inputs = prepareDecode(ctx, seqs);
  }

  auto [input_ids, positions] = inputs;
  auto result = model_->predict(ctx, input_ids, positions);
  // auto v = result.value();
  // for (int i = 0; i < v.size(); ++i) {
  //   LOG(INFO) << i << ":Predicted token id: " << v[i];
  // }
  resetContext(ctx);
  return result;
}

size_t ModelRunner::getNumKVCacheBlocks() const { return num_kvcache_blocks_; }

}  // namespace ginfer::engine
