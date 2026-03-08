#include <glog/logging.h>
#include <tokenizers_cpp.h>
#include <vector>
#include "ginfer/core/model/model_factory.h"
#include "ginfer/core/model/qwen2.h"
#include "ginfer/core/model/tokenizer/auto_tokenizer.h"
#include "ginfer/core/op/op.h"
#include "ginfer/test/pybind/func_wrap.h"
#include "ginfer/test/pybind/test_registry.h"
#include "ginfer/test/pybind/type.h"
#include "ginfer/utils/utils.h"

namespace py = pybind11;

namespace ginfer::test::pybind {

using common::DeviceType;
using core::memory::Buffer;
using core::tensor::DataType;
using core::tensor::Shape;
using core::tensor::Tensor;
using core::tensor::TensorRef;

std::vector<int32_t> model_generate(const std::string& model_path,
                                    TensorRef input_ids,
                                    std::pair<int64_t, int64_t> pos_id_range) {
  auto loader = core::model::ModelFactory::createLoader(model_path);
  auto model = loader->load();
  auto to_device_res = model->toDevice(DeviceType::kDeviceCUDA);
  CHECK(to_device_res.ok()) << "Model toDevice failed: " << to_device_res.err();

  std::vector<int32_t> token_ids;

  // Move input tensors to CUDA (model is on CUDA in this test)
  auto input_ids_dev_res =
      Tensor::create(DataType::kDataTypeInt32, Shape({1}), DeviceType::kDeviceCUDA);
  CHECK(input_ids_dev_res.ok()) << input_ids_dev_res.err();
  auto input_ids_dev = input_ids_dev_res.value();
  input_ids->toDevice(DeviceType::kDeviceCUDA);

  // Prepare output tensor
  // prefill
  auto res = model->predict(input_ids, pos_id_range);
  CHECK(res.ok()) << "Model predict failed: " << res.err();
  int32_t next_token_id = res.value();
  input_ids->toDevice(DeviceType::kDeviceCPU);
  input_ids = input_ids->slice(0, 0, 1);
  input_ids->data<int32_t>()[0] = next_token_id;  // Update

  // decode loop
  while (!model->isEosToken(next_token_id)) {
    // LOG(INFO) << "Predicted token id: " << next_token_id;
    token_ids.push_back(next_token_id);
    pos_id_range = {pos_id_range.second + 1, pos_id_range.second + 1};
    input_ids_dev->copyFrom(*input_ids);
    auto res = model->predict(input_ids_dev, pos_id_range);
    CHECK(res.ok()) << "Model predict failed: " << res.err();
    next_token_id = res.value();
    input_ids->data<int32_t>()[0] = next_token_id;  // Update
  }

  token_ids.push_back(next_token_id);

  return token_ids;
}

TensorRef test_model_generate_cuda(const std::string& model_path,
                                   TensorRef input_ids,
                                   std::pair<int64_t, int64_t>& pos_id_range) {
  auto token_ids = model_generate(model_path, input_ids, pos_id_range);
  int64_t token_count = token_ids.size();

  auto buf_res = Buffer::create(token_count * sizeof(int32_t), DeviceType::kDeviceCPU);
  CHECK(buf_res.ok()) << buf_res.err();
  auto buf = buf_res.value();
  std::memcpy(buf->ptr(), token_ids.data(), buf->size());

  auto out_res = Tensor::create(DataType::kDataTypeInt32, Shape({token_count}), buf);
  CHECK(out_res.ok()) << out_res.err();
  return out_res.value();
}

std::string test_model_infer_cuda(const std::string& model_path, const std::string& prompt) {
  // Tokenize prompt
  auto tokenizer = std::make_unique<core::model::tokenizer::AutoTokenizer>(model_path);
  auto conversation = nlohmann::json::array({{{"role", "user"}, {"content", prompt}}});
  auto input_content = tokenizer->applyChatTemplate(conversation);
  auto input_ids_vec = tokenizer->encode(input_content);

  auto buf_res = Buffer::create(input_ids_vec.size() * sizeof(int32_t), DeviceType::kDeviceCPU);
  CHECK(buf_res.ok()) << buf_res.err();
  auto buf = buf_res.value();
  std::memcpy(buf->ptr(), input_ids_vec.data(), buf->size());

  auto input_ids_res = Tensor::create(DataType::kDataTypeInt32,
                                      Shape({static_cast<int64_t>(input_ids_vec.size())}), buf);
  CHECK(input_ids_res.ok()) << input_ids_res.err();
  auto input_ids = input_ids_res.value();

  std::pair<int64_t, int64_t> pos_id_range{0, static_cast<int64_t>(input_ids_vec.size()) - 1};
  auto next_token_ids = model_generate(model_path, input_ids, pos_id_range);
  auto all_token_ids = input_ids_vec;
  all_token_ids.insert(all_token_ids.end(), next_token_ids.begin(), next_token_ids.end());
  return tokenizer->decode(all_token_ids, true);
}

REGISTER_PYBIND_TEST(test_model_generate_cuda);
REGISTER_PYBIND_TEST(test_model_infer_cuda);

}  // namespace ginfer::test::pybind
