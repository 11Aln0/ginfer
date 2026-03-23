#include "ginfer/core/op/kernels/kernel_registry.h"
#include "ginfer/core/op/kernels/rope_kernel.h"
#include "ginfer/core/op/op.h"

namespace ginfer::core::op {

RotaryEmbeddingOp::RotaryEmbeddingOp(DeviceType dev_type, float rope_theta)
    : AutoKernelDispatchOp<kernel::RotaryEmbeddingKernelFuncType>(
          dev_type, OpType::kOpCustom, "rotaryEmbedding"),
      rope_theta_(rope_theta) {}

Result<void, std::string> RotaryEmbeddingOp::run(const core::InferContext& ctx,
                                                 const std::vector<const Tensor*>& inputs,
                                                 std::vector<Tensor*> outputs) {
  CHECK(inputs.size() == 1) << "RotaryEmbeddingOp requires exactly 1 input tensor.";
  CHECK(outputs.size() == 2) << "RotaryEmbeddingOp requires exactly 2 output tensors.";

  common::DeviceType dev_type = getDeviceType();
  auto pos_ids_range = inputs[0]->data<int64_t>();
  Tensor* sin_cache = outputs[0];
  Tensor* cos_cache = outputs[1];

  auto kernel = getKernel(dev_type, sin_cache->dtype());
  auto dev_ctx = common::DeviceContext::create(dev_type);
  kernel(*dev_ctx, *sin_cache, *cos_cache, pos_ids_range[0], pos_ids_range[1], rope_theta_);

  return Ok<void>();
}

Llama3RotaryEmbeddingOp::Llama3RotaryEmbeddingOp(DeviceType dev_type,
                                                 float rope_theta,
                                                 float factor,
                                                 float high_freq_factor,
                                                 float low_freq_factor,
                                                 int old_ctx_len)
    : AutoKernelDispatchOp<kernel::Llama3RotaryEmbeddingKernelFuncType>(
          dev_type, OpType::kOpCustom, "llama3RotaryEmbedding"),
      rope_theta_(rope_theta),
      factor_(factor),
      high_freq_factor_(high_freq_factor),
      low_freq_factor_(low_freq_factor),
      old_ctx_len_(old_ctx_len) {}

Result<void, std::string> Llama3RotaryEmbeddingOp::run(const core::InferContext& ctx,
                                                       const std::vector<const Tensor*>& inputs,
                                                       std::vector<Tensor*> outputs) {
  CHECK(inputs.size() == 1) << "Llama3RotaryEmbeddingOp requires exactly 1 input tensor.";
  CHECK(outputs.size() == 2) << "Llama3RotaryEmbeddingOp requires exactly 2 output tensors.";

  common::DeviceType dev_type = getDeviceType();
  auto pos_ids_range = inputs[0]->data<int64_t>();
  Tensor* sin_cache = outputs[0];
  Tensor* cos_cache = outputs[1];

  auto kernel = getKernel(dev_type, sin_cache->dtype());
  auto dev_ctx = common::DeviceContext::create(dev_type);
  kernel(*dev_ctx, *sin_cache, *cos_cache, pos_ids_range[0], pos_ids_range[1], rope_theta_, factor_,
         high_freq_factor_, low_freq_factor_, old_ctx_len_);

  return Ok<void>();
}

ROPEOp::ROPEOp(DeviceType dev_type)
    : AutoKernelDispatchOp<kernel::ROPEKernelFuncType>(dev_type, OpType::kOpROPE, "ROPE") {}

Result<void, std::string> ROPEOp::run(const core::InferContext& ctx,
                                      const std::vector<const Tensor*>& inputs,
                                      std::vector<Tensor*> outputs) {
  CHECK(inputs.size() == 4)
      << "ROPEOp requires exactly 4 input tensors (input, positions, sin_cache, cos_cache).";
  CHECK(outputs.size() == 1) << "ROPEOp requires exactly 1 output tensor.";

  const Tensor* input = inputs[0];      // [seq_len, num_heads or num_kv_heads, head_dim]
  const Tensor* positions = inputs[1];  // [seq_len]
  const Tensor* sin_cache = inputs[2];
  const Tensor* cos_cache = inputs[3];

  common::DeviceType dev_type = getDeviceType();

  auto kernel = getKernel(dev_type, input->dtype());
  auto dev_ctx = common::DeviceContext::create(dev_type);
  kernel(*dev_ctx, *input, *positions, *sin_cache, *cos_cache, *outputs[0]);

  return Ok<void>();
}

}  // namespace ginfer::core::op
