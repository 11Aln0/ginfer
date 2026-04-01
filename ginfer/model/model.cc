#include "ginfer/model/model.h"
#include "ginfer/core/context.h"
#include "ginfer/core/memory/allocator_factory.h"

namespace ginfer::model {

// Model
Model::Model(ModelConfig config, common::DeviceType dev_type)
    : config_(std::move(config)), dev_type_(dev_type) {}

Result<void, std::string> Model::toDevice(common::DeviceType dev_type) {
  dev_type_ = dev_type;
  return Ok<void>();
}

common::DeviceType Model::getDeviceType() const { return dev_type_; }

const ModelConfig& Model::getConfig() const { return config_; }

bool Model::isEosToken(int32_t token_id) const {
  for (auto id : config_.eos_token_ids) {
    if (id == token_id) return true;
  }
  return false;
}

void Model::setRuntimeConfig(ModelRuntimeConfig runtime_config) {
  runtime_config_ = std::move(runtime_config);
}

// end Model

// LlamaArchModel
LlamaArchModel::LlamaArchModel(LlamaArchModelConfig config, common::DeviceType dev_type)
    : config_(config),
      embed_tokens(dev_type, "embed_tokens"),
      final_rmsnorm(dev_type, "final_rmsnorm", config.rms_norm_eps),
      lm_head(dev_type, "lm_head"),
      argmax_op(dev_type),
      Model(config, dev_type) {
  // initialize encoder layers
  for (size_t i = 0; i < config.nlayer; ++i) {
    encoder_layers.emplace_back(dev_type, "encoder_layer_" + std::to_string(i), config.rms_norm_eps,
                                config.num_heads, config.num_kv_heads, config.head_dim);
  }
}

Result<void, std::string> LlamaArchModel::lazyAllocIntermediates() {
  if (intermediates_allocated_) return Ok<void>();

  using core::memory::DeviceType;
  using core::tensor::DataType;
  using core::tensor::Shape;
  using core::tensor::Tensor;

  int max_position_embeddings = config_.max_position_embeddings;
  int max_seq_len = runtime_config_.max_seq_len;
  int head_dim = config_.head_dim;
  auto dtype = config_.dtype;
  auto dev_type = dev_type_;

  auto& i = intermediates_;
  DECLARE_OR_RETURN(t0, Tensor::create(dtype, Shape{max_seq_len, config_.hidden_size}, dev_type));
  ASSIGN_OR_RETURN(i.embed_out,
                   Tensor::create(dtype, Shape{max_seq_len, config_.hidden_size}, dev_type));
  ASSIGN_OR_RETURN(
      i.lm_head_out,
      Tensor::create(dtype, Shape{runtime_config_.max_batch_size, config_.vocab_size}, dev_type));
  ASSIGN_OR_RETURN(i.argmax_out, Tensor::create(DataType::kDataTypeInt32,
                                                Shape{runtime_config_.max_batch_size}, dev_type));
  i.norm_out = t0;

  DECLARE_OR_RETURN(
      last_hidden_state,
      Tensor::create(dtype, Shape{runtime_config_.max_batch_size, config_.hidden_size}, dev_type));
  core::layer::transformer::LMHeadLayer::Intermediates lm_head_intermediates = {
      .last_hidden_state = last_hidden_state,
  };

  DECLARE_OR_RETURN(
      t1, Tensor::create(dtype, Shape{max_seq_len, config_.num_heads * head_dim}, dev_type));
  DECLARE_OR_RETURN(
      t_k, Tensor::create(dtype, Shape{max_seq_len, config_.num_kv_heads * head_dim}, dev_type));
  DECLARE_OR_RETURN(
      t_v, Tensor::create(dtype, Shape{max_seq_len, config_.num_kv_heads * head_dim}, dev_type));

  core::layer::transformer::AttentionLayer::Intermediates attn_intermediates = {
      .q_proj_out = t1,
      .k_proj_out = t_k,
      .v_proj_out = t_v,
      .gqa_out = t1,
  };

  DECLARE_OR_RETURN(t2,
                    Tensor::create(dtype, Shape{max_seq_len, config_.intermediate_size}, dev_type));
  DECLARE_OR_RETURN(up_out,
                    Tensor::create(dtype, Shape{max_seq_len, config_.intermediate_size}, dev_type));
  core::layer::transformer::FeedForwardLayer::Intermediates mlp_intermediates = {
      .gate_out = t2,
      .up_out = up_out,
      .swiglu_out = t2,
  };

  DECLARE_OR_RETURN(attn_out,
                    Tensor::create(dtype, Shape{max_seq_len, config_.hidden_size}, dev_type));
  core::layer::transformer::EncoderLayer::Intermediates encoder_intermediates = {
      .attn = attn_intermediates,
      .mlp = mlp_intermediates,
      .norm_out = t0,
      .attn_out = attn_out,
  };

  for (auto& e : encoder_layers) {
    e.setIntermediates(encoder_intermediates);
  }
  lm_head.setIntermediates(lm_head_intermediates);

  intermediates_allocated_ = true;
  return Ok<void>();
}

void LlamaArchModel::setKVCache(int layer_id, TensorRef& k_cache, TensorRef& v_cache) {
  CHECK_THROW(layer_id >= 0 && layer_id < config_.nlayer, "Invalid layer id: {}", layer_id);
  // [num_kvcache_blocks, block_size, num_kv_heads, head_dim]
  CHECK_THROW(k_cache->shape().ndim() == 4, "Invalid k_cache shape, expected 4D but got {}",
              k_cache->shape().ndim());
  encoder_layers[layer_id].getAttentionLayer().setKVCache(k_cache, v_cache);
}

Result<void, std::string> LlamaArchModel::lazyInitPosEmbedding(const core::InferContext& ctx) {
  using core::memory::DeviceType;
  using core::tensor::DataType;
  using core::tensor::Shape;
  using core::tensor::Tensor;

  if (pos_embedding_initialized_) return Ok<void>();
  ASSIGN_OR_RETURN(
      sin, Tensor::create(DataType::kDataTypeFloat32,
                          Shape{config_.max_position_embeddings, config_.head_dim / 2}, dev_type_));
  ASSIGN_OR_RETURN(
      cos, Tensor::create(DataType::kDataTypeFloat32,
                          Shape{config_.max_position_embeddings, config_.head_dim / 2}, dev_type_));
  auto allocator = core::memory::getDeviceAllocator(DeviceType::kDeviceCPU, core::memory::kPooled);
  DECLARE_OR_RETURN(pos_ids, Tensor::create(DataType::kDataTypeInt64, Shape{2}, allocator));
  pos_ids->data<int64_t>()[0] = 0;
  pos_ids->data<int64_t>()[1] = config_.max_position_embeddings - 1;
  RETURN_ON_ERR(getRotaryEmbeddingOp().run(ctx, {pos_ids.get()}, {sin.get(), cos.get()}));
  pos_embedding_initialized_ = true;
  return Ok<void>();
}

Result<void, std::string> LlamaArchModel::forward(const core::InferContext& ctx,
                                                  const TensorRef input_ids,
                                                  const TensorRef positions,
                                                  TensorRef output) {
  int64_t seq_len = input_ids->shape()[0];

  auto embed_out = intermediates_.embed_out->slice(0, 0, seq_len);
  RETURN_ON_ERR(embed_tokens.forward(ctx, {input_ids}, embed_out));
  TensorRef hidden_state = embed_out;
  for (int i = 0; i < config_.nlayer; ++i) {
    auto& encoder = encoder_layers[i];
    RETURN_ON_ERR(encoder.forward(ctx, {hidden_state, positions, sin, cos}, hidden_state));
  }
  return final_rmsnorm.forward(ctx, {hidden_state}, output);
}

Result<void, std::string> LlamaArchModel::toDevice(common::DeviceType dev_type) {
  dev_type_ = dev_type;
  RETURN_ON_ERR(embed_tokens.toDevice(dev_type));
  RETURN_ON_ERR(getRotaryEmbeddingOp().toDevice(dev_type));
  for (auto& encoder : encoder_layers) {
    RETURN_ON_ERR(encoder.toDevice(dev_type));
  }
  RETURN_ON_ERR(final_rmsnorm.toDevice(dev_type));
  RETURN_ON_ERR(lm_head.toDevice(dev_type));
  return argmax_op.toDevice(dev_type);
}

Result<std::vector<int32_t>, std::string> LlamaArchModel::predict(
    const core::InferContext& ctx,
    const core::tensor::TensorRef token_ids,
    const core::tensor::TensorRef positions) {
  using core::tensor::DataType;

  RETURN_ON_ERR(lazyAllocIntermediates());
  RETURN_ON_ERR(lazyInitPosEmbedding(ctx));

  auto& rt_cfg = runtime_config_;
  int64_t seq_len = token_ids->shape()[0];

  CHECK(ctx.dev_ctx.value()->getDeviceType() == dev_type_ && token_ids->devType() == dev_type_)
      << "Device type mismatch between InferContext and input tensors.";
  CHECK_LE(seq_len, static_cast<int64_t>(rt_cfg.max_seq_len))
      << "Input token_ids length exceeds max_seq_len.";
  CHECK_EQ(positions->shape()[0], seq_len) << "positions length must equal input token_ids length.";
  CHECK(token_ids->dtype() == DataType::kDataTypeInt32) << "token_ids dtype must be int32.";

  int batch_size = 1;
  if (ctx.block_tables.has_value()) {
    CHECK(ctx.cu_seqlens_kv.has_value()) << "cu_seqlens_kv are required in InferContext.";
    if (ctx.is_prefill) {
      CHECK(ctx.cu_seqlens_q.has_value()) << "cu_seqlens_q are required in InferContext.";
    }
    batch_size = ctx.block_tables.value()->shape()[0];
  }

  CHECK_LE(batch_size, rt_cfg.max_batch_size) << "Batch size exceeds max_batch_size.";

  // forward
  auto norm_out = intermediates_.norm_out->slice(0, 0, seq_len);
  auto lm_head_out = intermediates_.lm_head_out->slice(0, 0, batch_size);
  RETURN_ON_ERR(forward(ctx, token_ids, positions, norm_out));

  // [total_tokens, vocab_size] -> [batch_size, vocab_size]
  RETURN_ON_ERR(lm_head.forward(ctx, {norm_out}, lm_head_out));

  // get next token id (argmax)
  auto argmax_out = intermediates_.argmax_out->slice(0, 0, batch_size);
  RETURN_ON_ERR(argmax_op.run(ctx, {lm_head_out.get()}, {argmax_out.get()}));
  ASSIGN_OR_RETURN(argmax_out,
                   argmax_out->toDevice(common::DeviceType::kDeviceCPU, core::memory::kPooled));

  std::vector<int32_t> tokens(batch_size);
  std::memcpy(tokens.data(), argmax_out->data(), batch_size * sizeof(int32_t));

  return Ok(std::move(tokens));
}

// end LlamaArchModel

}  // namespace ginfer::model
