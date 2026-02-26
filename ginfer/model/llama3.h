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

struct Llama3Config : public LlamaArchModelConfig {
  struct {
    float factor;
    float high_freq_factor;
    float low_freq_factor;
    int old_ctx_len;
  } rope_scaling;
};

class Llama3ModelLoader : public ModelLoader {
 public:
  explicit Llama3ModelLoader(std::string model_path);

  std::shared_ptr<Model> load() override;

 protected:
  ModelLoader::EncoderWeight loadEncoderLayerWeight(int layer_idx);

 private:
  Llama3Config loadConfig();
};

class Llama3Model : public LlamaArchModel {
 public:
  Llama3Model(Llama3Config config, common::DeviceType dev_type = common::DeviceType::kDeviceCPU);

 protected:
  op::Op& getRotaryEmbeddingOp() override { return rotary_emb; }
  friend class Llama3ModelLoader;

 private:
  op::Llama3RotaryEmbeddingOp rotary_emb;
};

};  // namespace ginfer::model