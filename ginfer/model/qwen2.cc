#include "ginfer/model/qwen2.h"

namespace ginfer::model {

Qwen2Model::Qwen2Model(Qwen2Config config, common::DeviceType dev_type)
    : config_(config),
      dev_type_(dev_type),
      embed_tokens(dev_type, "embed_tokens"),
      final_rmsnorm(dev_type, "final_rmsnorm", config.rms_norm_eps),
      lm_head(dev_type, "lm_head"),
      argmax_op(dev_type),
      rotary_emb(dev_type, config.rope_theta) {
  // initialize encoder layers
  for (size_t i = 0; i < config.nlayer; ++i) {
    encoder_layers.emplace_back(dev_type, "encoder_layer_" + std::to_string(i), config.rms_norm_eps, config.num_heads,
                                config.num_kv_heads, config.head_dim);
  }
  initKVCache();
}

void Qwen2Model::initKVCache() {
  kv_cache_.offset = 0;
  const auto& c = config_;
  int64_t max_seq_len = static_cast<int64_t>(c.max_seq_len);
  // init k,v cache
  // TODO lazy allocation
  for (int i = 0; i < config_.nlayer; ++i) {
    kv_cache_.k.push_back(std::make_shared<Tensor>(
        config_.dtype, tensor::Shape{max_seq_len, c.num_kv_heads, c.head_dim}, memory::DeviceType::kDeviceCPU));
    kv_cache_.v.push_back(std::make_shared<Tensor>(
        config_.dtype, tensor::Shape{max_seq_len, c.num_kv_heads, c.head_dim}, memory::DeviceType::kDeviceCPU));
  }
}

void Qwen2Model::lazyAllocIntermediates() {
  if (intermediates_allocated_) return;

  using memory::DeviceType;
  using tensor::DataType;
  using tensor::Shape;
  using tensor::Tensor;

  int64_t max_seq_len = static_cast<int64_t>(config_.max_seq_len);
  int64_t head_dim = static_cast<int64_t>(config_.head_dim);
  auto dtype = config_.dtype;
  auto dev_type = dev_type_;

  auto& i = intermediates_;
  auto t0 = std::make_shared<Tensor>(dtype, Shape{max_seq_len, config_.hidden_size}, dev_type);
  i.sin = std::make_shared<Tensor>(DataType::kDataTypeFloat32, Shape{max_seq_len, head_dim / 2}, dev_type);
  i.cos = std::make_shared<Tensor>(DataType::kDataTypeFloat32, Shape{max_seq_len, head_dim / 2}, dev_type);
  i.embed_out = std::make_shared<Tensor>(dtype, Shape{max_seq_len, config_.hidden_size}, dev_type);
  i.norm_out = t0;
  i.lm_head_out = std::make_shared<Tensor>(dtype, Shape{max_seq_len, config_.vocab_size}, dev_type);
  i.argmax_out = std::make_shared<Tensor>(DataType::kDataTypeInt64, Shape{1}, dev_type);

  layer::transformer::AttentionLayer::Intermediates attn_intermediates = {
      .q_proj_out = std::make_shared<Tensor>(dtype, Shape{max_seq_len, config_.num_heads * head_dim}, dev_type),
      .gqa_out = std::make_shared<Tensor>(dtype, Shape{max_seq_len, config_.num_heads * head_dim}, dev_type),
  };

  auto t1 = std::make_shared<Tensor>(dtype, Shape{max_seq_len, config_.intermediate_size}, dev_type);
  layer::transformer::FeedForwardLayer::Intermediates mlp_intermediates = {
      .gate_out = t1,
      .up_out = std::make_shared<Tensor>(dtype, Shape{max_seq_len, config_.intermediate_size}, dev_type),
      .swiglu_out = t1,
  };

  layer::transformer::EncoderLayer::Intermediates encoder_intermediates = {
      .attn = attn_intermediates,
      .mlp = mlp_intermediates,
      .norm_out = t0,
      .attn_out = std::make_shared<Tensor>(dtype, Shape{max_seq_len, config_.hidden_size}, dev_type),
  };

  for (auto& e : encoder_layers) {
    e.setIntermediates(encoder_intermediates);
  }

  intermediates_allocated_ = true;
}

std::pair<TensorRef, TensorRef> Qwen2Model::getPositionEmbedding(std::pair<int64_t, int64_t> pos_id_range) {
  auto [start, end] = pos_id_range;
  auto sin = intermediates_.sin->slice(0, 0, end - start + 1);
  auto cos = intermediates_.cos->slice(0, 0, end - start + 1);
  auto pos_ids = Tensor(tensor::DataType::kDataTypeInt64, tensor::Shape{2}, memory::DeviceType::kDeviceCPU);
  pos_ids.data<int64_t>()[0] = start;
  pos_ids.data<int64_t>()[1] = end;
  rotary_emb.run({&pos_ids}, {sin.get(), cos.get()});
  return {sin, cos};
}

Status Qwen2Model::forward(const TensorRef input_ids, std::pair<int64_t, int64_t> pos_id_range, TensorRef output) {
  int64_t seq_len = input_ids->shape()[0];
  auto [sin, cos] = getPositionEmbedding(pos_id_range);

  auto embed_out = intermediates_.embed_out->slice(0, 0, seq_len);
  RETURN_ON_ERROR(embed_tokens.forward({input_ids}, embed_out));
  TensorRef hidden_state = embed_out;
  for (int i = 0; i < config_.nlayer; ++i) {
    auto& encoder = encoder_layers[i];
    auto k_cache = kv_cache_.k[i]->slice(0, 0, kv_cache_.offset + seq_len);
    auto v_cache = kv_cache_.v[i]->slice(0, 0, kv_cache_.offset + seq_len);
    RETURN_ON_ERROR(encoder.forward({hidden_state, sin, cos, k_cache, v_cache}, hidden_state));
  }
  kv_cache_.offset += seq_len;
  return final_rmsnorm.forward({hidden_state}, output);
}

void Qwen2Model::toDevice(common::DeviceType dev_type) {
  dev_type_ = dev_type;
  embed_tokens.toDevice(dev_type);
  for (auto& encoder : encoder_layers) {
    encoder.toDevice(dev_type);
  }
  final_rmsnorm.toDevice(dev_type);
  lm_head.toDevice(dev_type);

  for (auto& k : kv_cache_.k) {
    k->toDevice(dev_type);
  }
  for (auto& v : kv_cache_.v) {
    v->toDevice(dev_type);
  }

  argmax_op.toDevice(dev_type);
  rotary_emb.toDevice(dev_type);
}

Status Qwen2Model::predict(const tensor::Tensor& token_ids, std::pair<int64_t, int64_t> pos_id_range,
                           int64_t& next_token_id) {
  lazyAllocIntermediates();

  int64_t seq_len = token_ids.shape()[0];
  CHECK_LE(seq_len, static_cast<int64_t>(config_.max_seq_len)) << "Input token_ids length exceeds max_seq_len.";
  auto input_ids = token_ids;
  input_ids.toDevice(InputTensorAllocator::getInstance());
  auto input_ids_ref = std::make_shared<tensor::Tensor>(input_ids);

  // forward
  auto norm_out = intermediates_.norm_out->slice(0, 0, seq_len);
  auto lm_head_out = intermediates_.lm_head_out->slice(0, 0, seq_len);
  RETURN_ON_ERROR(forward(input_ids_ref, pos_id_range, norm_out));
  RETURN_ON_ERROR(lm_head.forward({norm_out}, lm_head_out));

  // get next token id (argmax)
  lm_head_out = lm_head_out->slice(0, seq_len - 1, seq_len);  // only need the last token's logits
  auto argmax_out = intermediates_.argmax_out->slice(0, 0, 1);
  RETURN_ON_ERROR(argmax_op.run({lm_head_out.get()}, {argmax_out.get()}));
  argmax_out->toDevice(memory::DeviceType::kDeviceCPU);

  next_token_id = argmax_out->data<int64_t>()[0];

  return ginfer::error::Success();
}

}  // namespace ginfer::model
