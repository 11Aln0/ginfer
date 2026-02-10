#include <glog/logging.h>
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

int64_t test_qwen2_predict_cuda(const std::string& model_path, Tensor& input_ids,
                                std::pair<int64_t, int64_t>& pos_id_range) {
  model::Qwen2ModelLoader loader(model_path);
  auto model = loader.load();
  model->toDevice(DeviceType::kDeviceCUDA);

  // Move input tensors to CPU (model is on CPU in this test)
  input_ids.toDevice(DeviceType::kDeviceCUDA);

  // Prepare output tensor
  int64_t next_token_id = -1;

  // Run prediction
  auto status = model->predict(input_ids, pos_id_range, next_token_id);
  CHECK(status.code() == error::StatusCode::kSuccess) << "Qwen2Model predict failed: " << status.msg();

  return next_token_id;
}

}  // namespace ginfer::test::pybind