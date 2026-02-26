#pragma once

#include "ginfer/layer/layer.h"
#include "ginfer/op/op.h"
#include "ginfer/tensor/tensor.h"

namespace ginfer::layer::transformer {

class AttentionLayer : public Layer {
 public:
  struct Weight {
    TensorRef q_w;  // [hidden_size, num_heads * head_dim]
    TensorRef q_b;
    TensorRef k_w;  // [hidden_size, num_kv_heads * head_dim]
    TensorRef k_b;
    TensorRef v_w;  // [hidden_size, num_kv_heads * head_dim]
    TensorRef v_b;
    TensorRef o_w;  // [hidden_size, num_heads * head_dim]
    TensorRef o_b;
  };

  struct Intermediates {
    TensorRef q_proj_out;  // [max_seq_len, num_heads * head_dim]
    TensorRef gqa_out;     // [max_seq_len, num_heads * head_dim]
    // TensorRef o_proj_out;  // [max_seq_len, hidden_size]
  };

 public:
  AttentionLayer(DeviceType dev_type, std::string layer_name, int num_heads, int num_kv_heads, int head_dim);

  Result<void, std::string> forward(const std::vector<TensorRef>& inputs, TensorRef output) override;

  Result<void, std::string> toDevice(DeviceType dev_type) override;

  void setWeight(const Weight& weight);

  void setIntermediates(const Intermediates& intermediates);

 private:
  Intermediates intermediates_;

  op::ROPEOp rope_op;
  op::GQAOp gqa_op;

  LinearLayer q_proj;
  LinearLayer k_proj;
  LinearLayer v_proj;
  LinearLayer o_proj;

  int num_heads_;
  int num_kv_heads_;
  int head_dim_;
};

class FeedForwardLayer : public Layer {
 public:
  struct Weight {
    TensorRef gate_w;  // [intermediate_size, hidden_size]
    TensorRef up_w;    // [intermediate_size, hidden_size]
    TensorRef down_w;  // [hidden_size, intermediate_size]
  };

  struct Intermediates {
    TensorRef gate_out;    // [max_seq_len, intermediate_size]
    TensorRef up_out;      // [max_seq_len, intermediate_size]
    TensorRef swiglu_out;  // [max_seq_len, intermediate_size]
  };

 public:
  FeedForwardLayer(DeviceType dev_type, std::string layer_name);

  Result<void, std::string> forward(const std::vector<TensorRef>& inputs, TensorRef output) override;

  Result<void, std::string> toDevice(DeviceType dev_type) override;

  void setWeight(const Weight& weight);

  void setIntermediates(const Intermediates& intermediates);

 private:
  Intermediates intermediates_;

  layer::LinearLayer gate_proj;
  layer::LinearLayer up_proj;
  layer::LinearLayer down_proj;
  op::SwiGLUOp swiglu_op;
};

class EncoderLayer : public Layer {
 public:
  struct Weight {
    AttentionLayer::Weight attn;
    FeedForwardLayer::Weight mlp;
    TensorRef attn_norm;
    TensorRef mlp_norm;
  };

  struct Intermediates {
    AttentionLayer::Intermediates attn;
    FeedForwardLayer::Intermediates mlp;
    TensorRef norm_out;
    TensorRef attn_out;
  };

 public:
  EncoderLayer(DeviceType dev_type, std::string layer_name, float rms_norm_eps, int num_heads, int num_kv_heads,
               int head_dim);

  Result<void, std::string> forward(const std::vector<TensorRef>& inputs, TensorRef output) override;

  Result<void, std::string> toDevice(DeviceType dev_type) override;

  void setWeight(const Weight& weight);

  void setIntermediates(const Intermediates& intermediates);

  AttentionLayer& getAttentionLayer();

  FeedForwardLayer& getFeedForwardLayer();

  RMSNormLayer& getAttnNormLayer();

  RMSNormLayer& getMLPNormLayer();

 private:
  tensor::DataType dtype_;

  Intermediates intermediates_;

  op::AddOp add;

  RMSNormLayer mlp_norm;
  RMSNormLayer attn_norm;
  AttentionLayer self_attn;
  FeedForwardLayer mlp;
};

}  // namespace ginfer::layer::transformer