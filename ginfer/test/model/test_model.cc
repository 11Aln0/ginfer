#include <glog/logging.h>
#include <tokenizers_cpp.h>
#include <ranges>
#include <vector>
#include "ginfer/common/errors.h"
#include "ginfer/core/model/model_factory.h"
#include "ginfer/core/model/qwen2.h"
#include "ginfer/core/model/tokenizer/auto_tokenizer.h"
#include "ginfer/core/op/op.h"
#include "ginfer/test/pybind/func_wrap.h"
#include "ginfer/test/pybind/test_registry.h"
#include "ginfer/test/pybind/type.h"
#include "ginfer/utils/utils.h"

namespace py = pybind11;

namespace ginfer::test::pybind {

using common::DeviceType;
using core::memory::Buffer;
using core::memory::getDeviceAllocator;
using core::memory::PooledAllocStrategy;
using core::tensor::DataType;
using core::tensor::Shape;
using core::tensor::Tensor;
using core::tensor::TensorRef;

auto host_allocator = getDeviceAllocator<PooledAllocStrategy>(DeviceType::kDeviceCPU);
auto dev_allocator = getDeviceAllocator<PooledAllocStrategy>(DeviceType::kDeviceCUDA);

void setCuSeqlen(core::InferContext& ctx, int seqlen_q, int seqlen_kv) {
  DECLARE_OR_THROW(cu_seqlens_q,
                   Tensor::create(DataType::kDataTypeInt32, Shape({2}), host_allocator));
  DECLARE_OR_THROW(cu_seqlens_kv,
                   Tensor::create(DataType::kDataTypeInt32, Shape({2}), host_allocator));
  cu_seqlens_q->data<int32_t>()[0] = 0;
  cu_seqlens_q->data<int32_t>()[1] = seqlen_q;
  cu_seqlens_kv->data<int32_t>()[0] = 0;
  cu_seqlens_kv->data<int32_t>()[1] = seqlen_kv;
  ASSIGN_OR_THROW(cu_seqlens_q, cu_seqlens_q->toDevice(dev_allocator));
  ASSIGN_OR_THROW(cu_seqlens_kv, cu_seqlens_kv->toDevice(dev_allocator));
  ctx.cu_seqlens_q = cu_seqlens_q;
  ctx.cu_seqlens_kv = cu_seqlens_kv;
}

void setSlotMapping(core::InferContext& ctx, int num_slots, bool is_prefill) {
  TensorRef slot_mapping;
  if (is_prefill) {
    ASSIGN_OR_THROW(slot_mapping,
                    Tensor::create(DataType::kDataTypeInt32, Shape({num_slots}), host_allocator));
    for (int i : std::views::iota(0, num_slots)) {
      slot_mapping->data<int32_t>()[i] = i;
    }
  } else {
    ASSIGN_OR_THROW(slot_mapping,
                    Tensor::create(DataType::kDataTypeInt32, Shape({1}), host_allocator));
    slot_mapping->data<int32_t>()[0] = num_slots - 1;
  }

  ASSIGN_OR_THROW(slot_mapping, slot_mapping->toDevice(dev_allocator));
  ctx.slot_mapping = slot_mapping;
}

void setBlockTable(core::InferContext& ctx, int seq_lens, int block_size = 16) {
  int num_blocks = (seq_lens + block_size - 1) / block_size;
  DECLARE_OR_THROW(block_tables, Tensor::create(DataType::kDataTypeInt32, Shape({1, num_blocks}),
                                                host_allocator));
  for (int i : std::views::iota(0, num_blocks)) {
    block_tables->data<int32_t>()[i] = i;
  }
  ASSIGN_OR_THROW(block_tables, block_tables->toDevice(dev_allocator));
  ctx.block_tables = block_tables;
}

void setMaxSeqLenQ(core::InferContext& ctx, int seqlen_q) { ctx.max_seqlen_q = seqlen_q; }

void setInferCtx(
    core::InferContext& ctx, bool is_prefill, int seqlen_q, int seqlen_kv, int block_size) {
  auto dev_ctx = common::DeviceContext::create(DeviceType::kDeviceCUDA);
  ctx.setDeviceContext(dev_ctx);
  setCuSeqlen(ctx, seqlen_q, seqlen_kv);
  setSlotMapping(ctx, seqlen_kv, is_prefill);
  setBlockTable(ctx, seqlen_kv, block_size);
  setMaxSeqLenQ(ctx, seqlen_q);
}

void allocKVCache(std::shared_ptr<core::model::Model>& model,
                  int num_blocks,
                  int block_size,
                  int num_kv_heads,
                  int head_dim) {
  auto dtype = model->getDtype();
  for (int i = 0; i < model->getNumLayers(); ++i) {
    DECLARE_OR_THROW(k_cache,
                     Tensor::create(dtype, Shape({num_blocks, block_size, num_kv_heads, head_dim}),
                                    dev_allocator));

    DECLARE_OR_THROW(v_cache,
                     Tensor::create(dtype, Shape({num_blocks, block_size, num_kv_heads, head_dim}),
                                    dev_allocator));
    model->setKVCache(i, k_cache, v_cache);
  }
}

std::vector<int32_t> model_generate(const std::string& model_path, TensorRef input_ids) {
  const int block_size = 16;
  const int num_blocks = 4096 / block_size;  // support up to 4096 sequence length

  auto loader = core::model::ModelFactory::createLoader(model_path);
  auto model = loader->load();
  THROW_ON_ERR(model->toDevice(DeviceType::kDeviceCUDA));
  auto [num_heads, num_kv_heads, head_dim] = model->getAttentionConfig();
  allocKVCache(model, num_blocks, block_size, num_kv_heads, head_dim);

  std::vector<int32_t> new_token_ids;
  core::InferContext infer_ctx;

  auto next_positions = [&](TensorRef positions) -> TensorRef {
    ASSIGN_OR_THROW(positions, positions->toDevice(host_allocator));
    int len = positions->shape()[0];
    auto ret = positions->slice(0, len - 1, len);
    ret->data<int32_t>()[0]++;
    // LOG(INFO) << "Next position: " << ret->data<int32_t>()[0];
    ASSIGN_OR_THROW(ret, ret->toDevice(dev_allocator));
    return ret;
  };

  auto input_seqlen = input_ids->shape()[0];
  DECLARE_OR_THROW(positions,
                   Tensor::create(DataType::kDataTypeInt32, Shape({input_seqlen}), host_allocator));
  for (int i : std::views::iota(0, input_seqlen)) {
    positions->data<int32_t>()[i] = i;
  }
  ASSIGN_OR_THROW(positions, positions->toDevice(dev_allocator));

  // prefill
  setInferCtx(infer_ctx, /* is_prefill */ true, input_seqlen, input_seqlen, block_size);
  ASSIGN_OR_THROW(input_ids, input_ids->toDevice(dev_allocator));
  DECLARE_OR_THROW(next_token_ids, model->predict(infer_ctx, input_ids, positions));
  ASSIGN_OR_THROW(input_ids, input_ids->toDevice(host_allocator));
  input_ids = input_ids->slice(0, 0, 1);

  auto next_token_id = next_token_ids[0];
  input_ids->data<int32_t>()[0] = next_token_id;

  // decode loop
  while (!model->isEosToken(next_token_id)) {
    // LOG(INFO) << "Generated token id: " << next_token_id;
    new_token_ids.push_back(next_token_id);
    setInferCtx(infer_ctx, /* is_prefill */ false, 1, input_seqlen + new_token_ids.size(),
                block_size);
    ASSIGN_OR_THROW(input_ids, input_ids->toDevice(dev_allocator));
    positions = next_positions(positions);
    ASSIGN_OR_THROW(next_token_ids, model->predict(infer_ctx, input_ids, positions));
    ASSIGN_OR_THROW(input_ids, input_ids->toDevice(host_allocator));
    next_token_id = next_token_ids[0];
    input_ids->data<int32_t>()[0] = next_token_id;
  }
  new_token_ids.push_back(next_token_id);

  return new_token_ids;
}

TensorRef test_model_generate_cuda(const std::string& model_path, TensorRef input_ids) {
  auto token_ids = model_generate(model_path, input_ids);
  int64_t token_count = token_ids.size();

  DECLARE_OR_THROW(buf, Buffer::create(token_count * sizeof(int32_t), DeviceType::kDeviceCPU));
  std::memcpy(buf->ptr(), token_ids.data(), buf->size());

  DECLARE_OR_THROW(out, Tensor::create(DataType::kDataTypeInt32, Shape({token_count}), buf));
  return out;
}

std::string test_model_infer_cuda(const std::string& model_path, const std::string& prompt) {
  // Tokenize prompt
  auto tokenizer = std::make_unique<core::model::tokenizer::AutoTokenizer>(model_path);
  auto conversation = nlohmann::json::array({{{"role", "user"}, {"content", prompt}}});
  auto input_content = tokenizer->applyChatTemplate(conversation);
  auto input_ids_vec = tokenizer->encode(input_content);

  DECLARE_OR_THROW(buf,
                   Buffer::create(input_ids_vec.size() * sizeof(int32_t), DeviceType::kDeviceCPU));
  std::memcpy(buf->ptr(), input_ids_vec.data(), buf->size());

  DECLARE_OR_THROW(input_ids,
                   Tensor::create(DataType::kDataTypeInt32,
                                  Shape({static_cast<int64_t>(input_ids_vec.size())}), buf));

  auto next_token_ids = model_generate(model_path, input_ids);
  auto all_token_ids = input_ids_vec;
  all_token_ids.insert(all_token_ids.end(), next_token_ids.begin(), next_token_ids.end());
  return tokenizer->decode(all_token_ids, true);
}

REGISTER_PYBIND_TEST(test_model_generate_cuda);
REGISTER_PYBIND_TEST(test_model_infer_cuda);

}  // namespace ginfer::test::pybind
