#pragma once

#include <memory>
#include <nlohmann/json.hpp>
#include "ginfer/common/errors.h"
#include "ginfer/layer/layer.h"
#include "ginfer/layer/transformer/layer.h"
#include "ginfer/memory/allocator_factory.h"
#include "ginfer/model/loader/safetensor_loader.h"
#include "ginfer/model/model.h"
#include "ginfer/op/op.h"
#include "ginfer/tensor/tensor.h"

namespace ginfer::model {

class ModelLoader {
  using AttentionWeight = layer::transformer::AttentionLayer::Weight;
  using EncoderWeight = layer::transformer::EncoderLayer::Weight;
  using FeedForwardWeight = layer::transformer::FeedForwardLayer::Weight;

 public:
  explicit ModelLoader(std::string model_path);

  virtual std::shared_ptr<Model> load() = 0;

 protected:
  nlohmann::json loadConfigJSON();
  tensor::DataType parseDataType(const std::string& dtype_str) const;

  AttentionWeight loadAttentionWeight(const std::string& prefix);
  FeedForwardWeight loadFeedForwardWeight(const std::string& prefix);
  EncoderWeight loadEncoderLayerWeight(int layer_idx);

 protected:
  std::string model_path_;
  SafeTensorLoader weight_loader;
};

}  // namespace ginfer::model