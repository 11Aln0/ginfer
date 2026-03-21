#include "ginfer/core/model/model.h"
#include "ginfer/core/context.h"
#include "ginfer/core/memory/allocator_factory.h"

namespace ginfer::core::model {

// Model
Model::Model(ModelConfig config, common::DeviceType dev_type)
    : config_(std::move(config)), dev_type_(dev_type) {}

Result<void, std::string> Model::toDevice(common::DeviceType dev_type) {
  dev_type_ = dev_type;
  return Ok<void>();
}

int Model::getVocabSize() const { return config_.vocab_size; }

int Model::getNumLayers() const { return config_.nlayer; }

tensor::DataType Model::getDtype() const { return config_.dtype; }

std::tuple<int, int, int> Model::getAttentionConfig() const {
  return {config_.num_heads, config_.num_kv_heads, config_.head_dim};
}

bool Model::isEosToken(int32_t token_id) const {
  for (auto id : config_.eos_token_ids) {
    if (id == token_id) return true;
  }
  return false;
}

// end Model

// LlamaArchModel
LlamaArchModel::LlamaArchModel(LlamaArchModelConfig config, common::DeviceType dev_type)
    : config_(config),
      dev_type_(dev_type),
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

  using memory::DeviceType;
  using tensor::DataType;
  using tensor::Shape;
  using tensor::Tensor;

  int max_position_embeddings = config_.max_position_embeddings;
  int max_seq_len = config_.max_seq_len;
  int head_dim = config_.head_dim;
  auto dtype = config_.dtype;
  auto dev_type = dev_type_;

  auto& i = intermediates_;
  DECLARE_OR_RETURN(t0, Tensor::create(dtype, Shape{max_seq_len, config_.hidden_size}, dev_type));
  ASSIGN_OR_RETURN(i.embed_out,
                   Tensor::create(dtype, Shape{max_seq_len, config_.hidden_size}, dev_type));
  ASSIGN_OR_RETURN(i.lm_head_out, Tensor::create(dtype, Shape{1, config_.vocab_size}, dev_type));
  ASSIGN_OR_RETURN(i.argmax_out, Tensor::create(DataType::kDataTypeInt64, Shape{1}, dev_type));
  i.norm_out = t0;

  DECLARE_OR_RETURN(
      t1, Tensor::create(dtype, Shape{max_seq_len, config_.num_heads * head_dim}, dev_type));
  DECLARE_OR_RETURN(
      t_k, Tensor::create(dtype, Shape{max_seq_len, config_.num_kv_heads * head_dim}, dev_type));
  DECLARE_OR_RETURN(
      t_v, Tensor::create(dtype, Shape{max_seq_len, config_.num_kv_heads * head_dim}, dev_type));

  layer::transformer::AttentionLayer::Intermediates attn_intermediates = {
      .q_proj_out = t1,
      .k_proj_out = t_k,
      .v_proj_out = t_v,
      .gqa_out = t1,
  };

  DECLARE_OR_RETURN(t2,
                    Tensor::create(dtype, Shape{max_seq_len, config_.intermediate_size}, dev_type));
  DECLARE_OR_RETURN(up_out,
                    Tensor::create(dtype, Shape{max_seq_len, config_.intermediate_size}, dev_type));
  layer::transformer::FeedForwardLayer::Intermediates mlp_intermediates = {
      .gate_out = t2,
      .up_out = up_out,
      .swiglu_out = t2,
  };

  DECLARE_OR_RETURN(attn_out,
                    Tensor::create(dtype, Shape{max_seq_len, config_.hidden_size}, dev_type));
  layer::transformer::EncoderLayer::Intermediates encoder_intermediates = {
      .attn = attn_intermediates,
      .mlp = mlp_intermediates,
      .norm_out = t0,
      .attn_out = attn_out,
  };

  for (auto& e : encoder_layers) {
    e.setIntermediates(encoder_intermediates);
  }

  intermediates_allocated_ = true;
  return Ok<void>();
}

void LlamaArchModel::setKVCache(int layer_id, TensorRef& k_cache, TensorRef& v_cache) {
  CHECK_THROW(layer_id >= 0 && layer_id < config_.nlayer, "Invalid layer id: {}", layer_id);
  encoder_layers[layer_id].getAttentionLayer().setKVCache(k_cache, v_cache);
}

Result<void, std::string> LlamaArchModel::lazyInitPosEmbedding() {
  if (pos_embedding_initialized_) return Ok<void>();
  ASSIGN_OR_RETURN(
      sin, Tensor::create(tensor::DataType::kDataTypeFloat32,
                          tensor::Shape{config_.max_position_embeddings, config_.head_dim / 2},
                          dev_type_));
  ASSIGN_OR_RETURN(
      cos, Tensor::create(tensor::DataType::kDataTypeFloat32,
                          tensor::Shape{config_.max_position_embeddings, config_.head_dim / 2},
                          dev_type_));
  auto allocator =
      memory::getDeviceAllocator<memory::PooledAllocStrategy>(memory::DeviceType::kDeviceCPU);
  DECLARE_OR_RETURN(pos_ids,
                    Tensor::create(tensor::DataType::kDataTypeInt64, tensor::Shape{2}, allocator));
  pos_ids->data<int64_t>()[0] = 0;
  pos_ids->data<int64_t>()[1] = config_.max_position_embeddings - 1;
  getRotaryEmbeddingOp().run(core::InferContext{}, {pos_ids.get()}, {sin.get(), cos.get()});
  pos_embedding_initialized_ = true;
  return Ok<void>();
}

Result<void, std::string> LlamaArchModel::forward(const core::InferContext& infer_ctx,
                                                  const TensorRef input_ids,
                                                  const TensorRef positions,
                                                  TensorRef output) {
  int64_t seq_len = input_ids->shape()[0];

  auto embed_out = intermediates_.embed_out->slice(0, 0, seq_len);
  RETURN_ON_ERR(embed_tokens.forward(infer_ctx, {input_ids}, embed_out));
  TensorRef hidden_state = embed_out;
  for (int i = 0; i < config_.nlayer; ++i) {
    auto& encoder = encoder_layers[i];
    RETURN_ON_ERR(encoder.forward(infer_ctx, {hidden_state, positions, sin, cos}, hidden_state));
  }
  return final_rmsnorm.forward(infer_ctx, {hidden_state}, output);
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

Result<int32_t, std::string> LlamaArchModel::predict(const core::InferContext& infer_ctx,
                                                     const tensor::TensorRef token_ids,
                                                     const tensor::TensorRef positions) {
  RETURN_ON_ERR(lazyAllocIntermediates());
  RETURN_ON_ERR(lazyInitPosEmbedding());

  int64_t seq_len = token_ids->shape()[0];
  CHECK_LE(seq_len, static_cast<int64_t>(config_.max_seq_len))
      << "Input token_ids length exceeds max_seq_len.";
  CHECK(positions != nullptr) << "positions must not be null";
  CHECK_EQ(positions->shape()[0], seq_len) << "positions length must equal input token_ids length.";
  CHECK(token_ids->dtype() == tensor::DataType::kDataTypeInt32) << "token_ids dtype must be int32";

  // forward
  auto norm_out = intermediates_.norm_out->slice(0, 0, seq_len);
  auto lm_head_out = intermediates_.lm_head_out;
  RETURN_ON_ERR(forward(infer_ctx, token_ids, positions, norm_out));
  // only need the last token's hidden state for LM head
  norm_out = norm_out->slice(0, seq_len - 1, seq_len);
  RETURN_ON_ERR(lm_head.forward(infer_ctx, {norm_out}, lm_head_out));

  // get next token id (argmax)
  auto argmax_out = intermediates_.argmax_out->slice(0, 0, 1);
  RETURN_ON_ERR(argmax_op.run(infer_ctx, {lm_head_out.get()}, {argmax_out.get()}));
  ASSIGN_OR_RETURN(argmax_out,
                   argmax_out->toDevice(memory::getDeviceAllocator<memory::PooledAllocStrategy>(
                       memory::DeviceType::kDeviceCPU)));

  return Ok(static_cast<int32_t>(argmax_out->data<int64_t>()[0]));  // TODO argmax return int32_t
}

// end LlamaArchModel

}  // namespace ginfer::core::model
