#pragma once

#include <memory>
#include <tuple>
#include "ginfer/common/errors.h"
#include "ginfer/core/layer/layer.h"
#include "ginfer/core/layer/transformer/layer.h"
#include "ginfer/core/memory/allocator_factory.h"
#include "ginfer/core/op/op.h"
#include "ginfer/core/tensor/tensor.h"
#include "ginfer/model/loader/model_loader.h"
#include "ginfer/model/model.h"

namespace ginfer::model {

using ginfer::core::tensor::Tensor;
using ginfer::core::tensor::TensorRef;
using ginfer::model::LlamaArchModelLoader;
using ginfer::model::ModelLoader;

struct Llama3Config : public LlamaArchModelConfig {
  struct {
    float factor;
    float high_freq_factor;
    float low_freq_factor;
    int old_ctx_len;
  } rope_scaling;
};

class Llama3ModelLoader : public LlamaArchModelLoader {
 public:
  explicit Llama3ModelLoader(std::string model_path);

  std::unique_ptr<Model> load() override;

 protected:
  std::tuple<bool, bool, bool, bool> getAttentionBiasConfig() const override;

 private:
  Llama3Config loadConfig();
};

class Llama3Model : public LlamaArchModel {
 public:
  Llama3Model(Llama3Config config, common::DeviceType dev_type = common::DeviceType::kDeviceCPU);

 protected:
  core::op::Op& getRotaryEmbeddingOp() override { return rotary_emb; }
  friend class Llama3ModelLoader;

 private:
  core::op::Llama3RotaryEmbeddingOp rotary_emb;
};

};  // namespace ginfer::model