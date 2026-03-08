#include <glog/logging.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include "ginfer/common/context.h"
#include "ginfer/core/memory/allocator_factory.h"
#include "ginfer/core/op/op.h"
#include "ginfer/test/pybind/test_registry.h"
#include "ginfer/test/pybind/type.h"

namespace py = pybind11;

namespace ginfer::test::pybind {

using common::DeviceType;
using core::tensor::DataType;
using core::tensor::Shape;
using core::tensor::Tensor;
using core::tensor::TensorRef;

TensorRef test_rmsnorm_op_cuda(TensorRef input_tensor, TensorRef gamma_tensor, float epsilon) {
  auto out_res =
      Tensor::create(input_tensor->dtype(), Shape(input_tensor->shape()), DeviceType::kDeviceCPU);
  CHECK(out_res.ok()) << out_res.err();
  auto output_tensor = out_res.value();

  ::ginfer::core::op::RMSNormOp rmsnorm_op(DeviceType::kDeviceCUDA, epsilon);

  // Move tensors to GPU
  auto cu_allocator =
      ginfer::core::memory::getDeviceAllocator<ginfer::core::memory::PooledAllocStrategy>(
          DeviceType::kDeviceCUDA);
  input_tensor->toDevice(cu_allocator);
  gamma_tensor->toDevice(cu_allocator);
  output_tensor->toDevice(cu_allocator);

  // Run computation
  std::vector<const Tensor*> inputs = {input_tensor.get(), gamma_tensor.get()};
  std::vector<Tensor*> outputs = {output_tensor.get()};
  auto status = rmsnorm_op.run(common::InferContext{}, inputs, outputs);
  CHECK(status.ok()) << "RMSNormOp run failed: " << status.err();

  // Copy result back to CPU
  output_tensor->toDevice(DeviceType::kDeviceCPU);

  return output_tensor;
}

REGISTER_PYBIND_TEST(test_rmsnorm_op_cuda);

}  // namespace ginfer::test::pybind
