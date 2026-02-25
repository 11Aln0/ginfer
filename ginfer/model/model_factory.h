#pragma once

#include <memory>
#include <nlohmann/json.hpp>
#include "ginfer/common/errors.h"
#include "ginfer/model/loader/model_loader.h"
#include "ginfer/model/model.h"
#include "ginfer/model/qwen2.h"
#include "ginfer/utils/utils.h"

namespace ginfer::model {

class ModelFactory {
 public:
  static std::unique_ptr<ModelLoader> createLoader(const std::string& model_path) {
    auto json_str = utils::file::loadBytesFromFile(model_path + "/config.json");
    CHECK_THROW(json_str.ok(), "Failed to load config.json: {}", json_str.err());
    auto json = nlohmann::json::parse(json_str.value());
    std::string model_type = json.value("model_type", "");
    if (model_type == "qwen2")
      return std::make_unique<Qwen2ModelLoader>(model_path);
    else
      CHECK_THROW(false, "Unsupported model type: {}", model_type);
  }
};

}  // namespace ginfer::model