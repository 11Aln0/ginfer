#include <glog/logging.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include "ginfer/memory/allocator_factory.h"
#include "ginfer/op/op.h"
#include "ginfer/test/pybind/test_registry.h"
#include "ginfer/test/pybind/type.h"

namespace py = pybind11;

namespace ginfer::test::pybind {
using common::DeviceType;
using memory::Buffer;
using tensor::DataType;
using tensor::Shape;
using tensor::Tensor;

Tensor test_rmsnorm_op_cuda(Tensor& input_tensor, Tensor& gamma_tensor, float epsilon) {
  DataType dtype = input_tensor.dtype();
  const Shape& shape = input_tensor.shape();
  Tensor output_tensor(dtype, Shape(shape), DeviceType::kDeviceCPU);

  ::ginfer::op::RMSNormOp rmsnorm_op(DeviceType::kDeviceCUDA, epsilon);

  // Move tensors to GPU
  auto cu_allocator = ginfer::memory::GlobalCUDAAllocator<ginfer::memory::cuda::PooledAllocStrategy>::getInstance();
  input_tensor.toDevice(cu_allocator);
  gamma_tensor.toDevice(cu_allocator);
  output_tensor.toDevice(cu_allocator);

  // Run computation
  std::vector<const Tensor*> inputs = {&input_tensor, &gamma_tensor};
  std::vector<Tensor*> outputs = {&output_tensor};
  auto status = rmsnorm_op.run(inputs, outputs);
  CHECK(status.code() == ::ginfer::error::StatusCode::kSuccess) << "RMSNormOp run failed: " << status.msg();

  // Copy result back to CPU
  output_tensor.toDevice(DeviceType::kDeviceCPU);

  return output_tensor;
}

REGISTER_PYBIND_TEST(test_rmsnorm_op_cuda);

}  // namespace ginfer::test::pybind
