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

struct Qwen2Config : public LlamaArchModelConfig {};

class Qwen2ModelLoader : public LlamaArchModelLoader {
 public:
  explicit Qwen2ModelLoader(std::string model_path);

  std::unique_ptr<Model> load() override;

 protected:
  std::tuple<bool, bool, bool, bool> getAttentionBiasConfig() const override;

 private:
  Qwen2Config loadConfig();
};

class Qwen2Model : public LlamaArchModel {
 public:
  Qwen2Model(Qwen2Config config, common::DeviceType dev_type = common::DeviceType::kDeviceCPU);

 protected:
  core::op::Op& getRotaryEmbeddingOp() override { return rotary_emb; }
  friend class Qwen2ModelLoader;

 private:
  core::op::RotaryEmbeddingOp rotary_emb;
};

}  // namespace ginfer::model