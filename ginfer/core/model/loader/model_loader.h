#pragma once

#include <memory>
#include <nlohmann/json.hpp>
#include "ginfer/common/errors.h"
#include "ginfer/core/layer/layer.h"
#include "ginfer/core/layer/transformer/layer.h"
#include "ginfer/core/memory/allocator_factory.h"
#include "ginfer/core/model/loader/safetensor_loader.h"
#include "ginfer/core/model/model.h"
#include "ginfer/core/op/op.h"
#include "ginfer/core/tensor/tensor.h"

namespace ginfer::core::model {

class ModelLoader {
 public:
  explicit ModelLoader(std::string model_path);

  virtual std::unique_ptr<Model> load() = 0;

 protected:
  using AttentionWeight = layer::transformer::AttentionLayer::Weight;
  using EncoderWeight = layer::transformer::EncoderLayer::Weight;
  using FeedForwardWeight = layer::transformer::FeedForwardLayer::Weight;

 protected:
  nlohmann::json loadConfigJSON();
  tensor::DataType parseDataType(const std::string& dtype_str) const;

  AttentionWeight loadAttentionWeight(
      const std::string& prefix, bool q_bias, bool k_bias, bool v_bias, bool o_bias);
  FeedForwardWeight loadFeedForwardWeight(const std::string& prefix);
  EncoderWeight loadEncoderLayerWeight(int layer_idx);

 protected:
  std::string model_path_;
  SafeTensorLoader weight_loader;
};

}  // namespace ginfer::core::model