#pragma once

#include <memory>
#include <string>
#include <vector>

#include "ginfer/common/device.h"
#include "ginfer/common/errors.h"
#include "ginfer/core/context.h"
#include "ginfer/core/op/kernels/kernel_dispatcher.h"
#include "ginfer/core/op/kernels/kernels.h"
#include "ginfer/core/op/kernels/page_attn_kernel.h"
#include "ginfer/core/op/kernels/rope_kernel.h"
#include "ginfer/core/tensor/tensor.h"

namespace ginfer::core::op {

using ginfer::common::DeviceType;
using ginfer::core::tensor::Tensor;

enum class OpType : uint8_t {
  kOpUnknown = 0,
  kOpLinear = 1,
  kOpEncode = 2,
  kOpEmbedding = 3,
  kOpRMSNorm = 4,
  kOpMatmul = 5,
  kOpROPE = 6,
  kOpMHA = 7,
  kOpGQA = 8,
  kOpSoftmax = 9,
  kOpAdd = 10,
  kOpSwiGLU = 11,
  kOpArgmax = 12,

  kOpCustom = 255,  // Custom ops start from this value
};

class BaseOp {
 public:
  explicit BaseOp(DeviceType dev_type, OpType op_type, std::string name);
  OpType opType() const;
  DeviceType getDeviceType() const;

  virtual Result<void, std::string> run(const core::InferContext& ctx,
                                        const std::vector<const Tensor*>& inputs,
                                        std::vector<Tensor*> outputs) = 0;
  virtual Result<void, std::string> toDevice(DeviceType dev_type);

 protected:
  const common::DeviceContext& getDeviceContext(const core::InferContext& ctx) const;

 private:
  DeviceType dev_type_ = DeviceType::kDeviceUnknown;
  OpType op_type_ = OpType::kOpUnknown;
  std::string name_ = "unknown";
};

class Op : public BaseOp {
  using BaseOp::BaseOp;
};

template <typename KernelFuncType>
class AutoKernelDispatchOp : public Op {
 public:
  AutoKernelDispatchOp(DeviceType dev_type,
                       OpType op_type,
                       std::string name,
                       std::string kernel_name)
      : dispatcher_(std::move(kernel_name)), Op(dev_type, op_type, std::move(name)) {}

  AutoKernelDispatchOp(DeviceType dev_type, OpType op_type, std::string name)
      : dispatcher_(name), Op(dev_type, op_type, name) {}

 protected:
  KernelFuncType getKernel(common::DeviceType dev_type,
                           tensor::DataType in_dtype,
                           tensor::DataType out_dtype) {
    return dispatcher_.getKernel(dev_type, in_dtype, out_dtype);
  }

  KernelFuncType getKernel(common::DeviceType dev_type, tensor::DataType dtype) {
    return dispatcher_.getKernel(dev_type, dtype);
  }

 private:
  kernel::KernelDispatcher<KernelFuncType> dispatcher_;
};

class MatmulOp : public Op {
 public:
  MatmulOp(DeviceType dev_type);

  virtual Result<void, std::string> run(const core::InferContext& ctx,
                                        const std::vector<const Tensor*>& inputs,
                                        std::vector<Tensor*> outputs) override;

 private:
  bool isGemvMode(const Tensor* A);

 private:
  kernel::KernelDispatcher<kernel::GemvKernelFuncType> gemv_dispatcher_{"gemv"};
  kernel::KernelDispatcher<kernel::GemmKernelFuncType> gemm_dispatcher_{"gemm"};
};

class AddOp : public AutoKernelDispatchOp<kernel::AddKernelFuncType> {
 public:
  AddOp(DeviceType dev_type);

  virtual Result<void, std::string> run(const core::InferContext& ctx,
                                        const std::vector<const Tensor*>& inputs,
                                        std::vector<Tensor*> outputs) override;

 private:
  bool checkBroadcastable(const tensor::Shape& shapeA, const tensor::Shape& shapeB);

  //  private:
  //   Result<void, std::string> checkParams(const std::vector<const Tensor*>& inputs, const
  //   std::vector<Tensor*> outputs);
};

class RMSNormOp : public AutoKernelDispatchOp<kernel::RMSNormKernelFuncType> {
 public:
  RMSNormOp(DeviceType dev_type, float epsilon);

  virtual Result<void, std::string> run(const core::InferContext& ctx,
                                        const std::vector<const Tensor*>& inputs,
                                        std::vector<Tensor*> outputs) override;

 private:
  float epsilon_;
};

class GQAOp : public AutoKernelDispatchOp<kernel::GQAKernelFuncType> {
 public:
  GQAOp(DeviceType dev_type);

  virtual Result<void, std::string> run(const core::InferContext& ctx,
                                        const std::vector<const Tensor*>& inputs,
                                        std::vector<Tensor*> outputs) override;

  // void setSeqLen(int seq_len);

 private:
  // int seq_len_;
};

class GQAVarlenOp : public AutoKernelDispatchOp<kernel::GQAVarlenKernelFuncType> {
 public:
  GQAVarlenOp(DeviceType dev_type, int paged_block_size);

  virtual Result<void, std::string> run(const core::InferContext& ctx,
                                        const std::vector<const Tensor*>& inputs,
                                        std::vector<Tensor*> outputs) override;

 private:
  int paged_block_size_;
};

class StoreKVCacheOp : public AutoKernelDispatchOp<kernel::StoreKVCacheKernelFuncType> {
 public:
  StoreKVCacheOp(DeviceType dev_type);

  virtual Result<void, std::string> run(const core::InferContext& ctx,
                                        const std::vector<const Tensor*>& inputs,
                                        std::vector<Tensor*> outputs) override;
};

class ArgmaxOp : public AutoKernelDispatchOp<kernel::ArgmaxKernelFuncType> {
 public:
  ArgmaxOp(DeviceType dev_type);

  virtual Result<void, std::string> run(const core::InferContext& ctx,
                                        const std::vector<const Tensor*>& inputs,
                                        std::vector<Tensor*> outputs) override;
};

class SelectLastTokenOp : public AutoKernelDispatchOp<kernel::SelectLastTokenKernelFuncType> {
 public:
  SelectLastTokenOp(DeviceType dev_type);

  virtual Result<void, std::string> run(const core::InferContext& ctx,
                                        const std::vector<const Tensor*>& inputs,
                                        std::vector<Tensor*> outputs) override;
};

class EmbeddingOp : public AutoKernelDispatchOp<kernel::EmbeddingKernelFuncType> {
 public:
  EmbeddingOp(DeviceType dev_type);

  virtual Result<void, std::string> run(const core::InferContext& ctx,
                                        const std::vector<const Tensor*>& inputs,
                                        std::vector<Tensor*> outputs) override;
};

class RotaryEmbeddingOp : public AutoKernelDispatchOp<kernel::RotaryEmbeddingKernelFuncType> {
 public:
  RotaryEmbeddingOp(DeviceType dev_type, float rope_theta = 10000.0f);

  virtual Result<void, std::string> run(const core::InferContext& ctx,
                                        const std::vector<const Tensor*>& inputs,
                                        std::vector<Tensor*> outputs) override;

 private:
  float rope_theta_;
};

class Llama3RotaryEmbeddingOp
    : public AutoKernelDispatchOp<kernel::Llama3RotaryEmbeddingKernelFuncType> {
 public:
  Llama3RotaryEmbeddingOp(DeviceType dev_type,
                          float rope_theta,
                          float factor,
                          float high_freq_factor,
                          float low_freq_factor,
                          int old_ctx_len);

  virtual Result<void, std::string> run(const core::InferContext& ctx,
                                        const std::vector<const Tensor*>& inputs,
                                        std::vector<Tensor*> outputs) override;

 private:
  float rope_theta_;
  float factor_;
  float high_freq_factor_;
  float low_freq_factor_;
  int old_ctx_len_;
};

class ROPEOp : public AutoKernelDispatchOp<kernel::ROPEKernelFuncType> {
 public:
  ROPEOp(DeviceType dev_type);

  virtual Result<void, std::string> run(const core::InferContext& ctx,
                                        const std::vector<const Tensor*>& inputs,
                                        std::vector<Tensor*> outputs) override;

  // void updateCache(int start_pos, int end_pos);

 private:
  // int head_dim_;
  // int max_seq_len_;
  // float rope_theta_;
  // std::shared_ptr<Tensor> sin_cache_;
  // std::shared_ptr<Tensor> cos_cache_;
};

class SwiGLUOp : public AutoKernelDispatchOp<kernel::SwiGluKernelFuncType> {
 public:
  SwiGLUOp(DeviceType dev_type);

  virtual Result<void, std::string> run(const core::InferContext& ctx,
                                        const std::vector<const Tensor*>& inputs,
                                        std::vector<Tensor*> outputs) override;
};

}  // namespace ginfer::core::op
