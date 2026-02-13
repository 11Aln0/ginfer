#include <glog/logging.h>
#include <vector>
#include "ginfer/model/qwen2.h"
#include "ginfer/op/op.h"
#include "ginfer/test/pybind/func_wrap.h"
#include "ginfer/test/pybind/test_registry.h"
#include "ginfer/test/pybind/type.h"

namespace py = pybind11;

namespace ginfer::test::pybind {

using common::DeviceType;
using memory::Buffer;
using tensor::DataType;
using tensor::Shape;
using tensor::Tensor;

Tensor test_qwen2_generate_cuda(const std::string& model_path, Tensor& input_ids,
                                std::pair<int64_t, int64_t>& pos_id_range) {
  model::Qwen2ModelLoader loader(model_path);
  auto model = loader.load();
  model->toDevice(DeviceType::kDeviceCUDA);

  std::vector<int64_t> token_ids;

  // Move input tensors to CUDA (model is on CUDA in this test)
  auto input_ids_dev = Tensor(DataType::kDataTypeInt64, Shape({1}), DeviceType::kDeviceCUDA);
  input_ids.toDevice(DeviceType::kDeviceCUDA);

  // Prepare output tensor
  int64_t next_token_id = -1;
  // prefill
  auto status = model->predict(input_ids, pos_id_range, next_token_id);
  CHECK(status.code() == error::StatusCode::kSuccess) << "Qwen2Model predict failed: " << status.msg();
  input_ids.toDevice(DeviceType::kDeviceCPU);
  input_ids = *input_ids.slice(0, 0, 1);
  input_ids.data<int64_t>()[0] = next_token_id;  // Update

  // decode loop
  while (next_token_id != 151643) {
    // LOG(INFO) << "Predicted token id: " << next_token_id;
    token_ids.push_back(next_token_id);
    pos_id_range = {pos_id_range.second + 1, pos_id_range.second + 1};
    input_ids_dev.copyFrom(input_ids);
    status = model->predict(input_ids_dev, pos_id_range, next_token_id);
    CHECK(status.code() == error::StatusCode::kSuccess) << "Qwen2Model predict failed: " << status.msg();
    input_ids.data<int64_t>()[0] = next_token_id;  // Update
  }

  token_ids.push_back(next_token_id);

  int64_t token_count = token_ids.size();

  auto buf = std::make_shared<Buffer>(token_count * sizeof(int64_t), DeviceType::kDeviceCPU);
  std::memcpy(buf->ptr(), token_ids.data(), buf->size());

  return Tensor(DataType::kDataTypeInt64, Shape({token_count}), buf);
}

REGISTER_PYBIND_TEST(test_qwen2_generate_cuda);

}  // namespace ginfer::test::pybind