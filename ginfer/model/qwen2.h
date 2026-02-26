#pragma once

#include <memory>
#include "ginfer/common/errors.h"
#include "ginfer/layer/layer.h"
#include "ginfer/layer/transformer/layer.h"
#include "ginfer/memory/allocator_factory.h"
#include "ginfer/model/loader/model_loader.h"
#include "ginfer/model/model.h"
#include "ginfer/op/op.h"
#include "ginfer/tensor/tensor.h"

namespace ginfer::model {

using ginfer::model::ModelLoader;
using ginfer::tensor::Tensor;
using ginfer::tensor::TensorRef;

struct Qwen2Config : public LlamaArchModelConfig {};

class Qwen2ModelLoader : public ModelLoader {
 public:
  explicit Qwen2ModelLoader(std::string model_path);

  std::shared_ptr<Model> load() override;

 protected:
  ModelLoader::EncoderWeight loadEncoderLayerWeight(int layer_idx);

 private:
  Qwen2Config loadConfig();
};

class Qwen2Model : public LlamaArchModel {
 public:
  Qwen2Model(Qwen2Config config, common::DeviceType dev_type = common::DeviceType::kDeviceCPU);

 protected:
  op::Op& getRotaryEmbeddingOp() override { return rotary_emb; }
  friend class Qwen2ModelLoader;

 private:
  op::RotaryEmbeddingOp rotary_emb;
};

}  // namespace ginfer::model