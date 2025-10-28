#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include "ginfer/tensor/tensor.h"

TEST(TensorTest, fromBuffer) {
  // cpu test

  auto cpu_allocator = ginfer::memory::CPUAllocatorFactory::get_instance();
  ASSERT_NE(cpu_allocator, nullptr);
  float* ptr1 = new float[128];
  auto cpu_buffer = std::make_shared<ginfer::memory::Buffer>(
      128 * sizeof(float), ptr1, ginfer::memory::DeviceType::kDeviceCPU);
  ginfer::tensor::Shape shape{32, 4};
  ginfer::tensor::Tensor cpu_tensor =
      ginfer::tensor::Tensor(ginfer::tensor::Dtype::kDtypeFloat32, shape, cpu_buffer);
  ASSERT_EQ(cpu_tensor.dtype(), ginfer::tensor::Dtype::kDtypeFloat32);
  ASSERT_EQ(cpu_tensor.shape().ndim(), 2);
  ASSERT_EQ(cpu_tensor.shape()[0], 32);
  ASSERT_EQ(cpu_tensor.shape()[1], 4);
  ASSERT_EQ(cpu_tensor.size(), 128);
  ASSERT_EQ(cpu_tensor.nbytes(), 128 * sizeof(float));
  auto strides = cpu_tensor.strides();
  ASSERT_EQ(strides.size(), 2);
  ASSERT_EQ(strides[0], 4);
  ASSERT_EQ(strides[1], 1);
  delete[] ptr1;  // Buffer should not free external memory

  // cuda test

  auto cuda_allocator = ginfer::memory::CUDAAllocatorFactory::get_instance();
  ASSERT_NE(cuda_allocator, nullptr);
  float* ptr2 = nullptr;
  cudaError_t err = cudaMallocManaged(reinterpret_cast<void**>(&ptr2), 128 * sizeof(float));
  ASSERT_EQ(err, cudaSuccess);
  auto cuda_buffer = std::make_shared<ginfer::memory::Buffer>(
      128 * sizeof(float), ptr2, ginfer::memory::DeviceType::kDeviceCUDA);
  ginfer::tensor::Tensor cuda_tensor =
      ginfer::tensor::Tensor(ginfer::tensor::Dtype::kDtypeFloat32, shape, cuda_buffer);
  ASSERT_EQ(cuda_tensor.dtype(), ginfer::tensor::Dtype::kDtypeFloat32);
  ASSERT_EQ(cuda_tensor.shape().ndim(), 2);
  ASSERT_EQ(cuda_tensor.shape()[0], 32);
  ASSERT_EQ(cuda_tensor.shape()[1], 4);
  ASSERT_EQ(cuda_tensor.size(), 128);
  ASSERT_EQ(cuda_tensor.nbytes(), 128 * sizeof(float));
  strides = cuda_tensor.strides();
  ASSERT_EQ(strides.size(), 2);
  ASSERT_EQ(strides[0], 4);
  ASSERT_EQ(strides[1], 1);
  cudaFree(ptr2);  // Buffer should not free external memory
}

TEST(TensorTest, fromAllocator) {
  // cpu test

  auto cpu_allocator = ginfer::memory::CPUAllocatorFactory::get_instance();
  ASSERT_NE(cpu_allocator, nullptr);
  ginfer::tensor::Shape shape{32, 4};
  ginfer::tensor::Tensor cpu_tensor = ginfer::tensor::Tensor(
      ginfer::tensor::Dtype::kDtypeFloat32, shape, ginfer::common::DeviceType::kDeviceCPU);
  ASSERT_EQ(cpu_tensor.dtype(), ginfer::tensor::Dtype::kDtypeFloat32);
  ASSERT_EQ(cpu_tensor.shape().ndim(), 2);
  ASSERT_EQ(cpu_tensor.shape()[0], 32);
  ASSERT_EQ(cpu_tensor.shape()[1], 4);
  ASSERT_EQ(cpu_tensor.size(), 128);
  ASSERT_EQ(cpu_tensor.nbytes(), 128 * sizeof(float));
  auto strides = cpu_tensor.strides();
  ASSERT_EQ(strides.size(), 2);
  ASSERT_EQ(strides[0], 4);
  ASSERT_EQ(strides[1], 1);

  // cuda test

  auto cuda_allocator = ginfer::memory::CUDAAllocatorFactory::get_instance();
  ASSERT_NE(cuda_allocator, nullptr);
  ginfer::tensor::Tensor cuda_tensor = ginfer::tensor::Tensor(
      ginfer::tensor::Dtype::kDtypeFloat32, shape, ginfer::common::DeviceType::kDeviceCUDA);
  ASSERT_EQ(cuda_tensor.dtype(), ginfer::tensor::Dtype::kDtypeFloat32);
  ASSERT_EQ(cuda_tensor.shape().ndim(), 2);
  ASSERT_EQ(cuda_tensor.shape()[0], 32);
  ASSERT_EQ(cuda_tensor.shape()[1], 4);
  ASSERT_EQ(cuda_tensor.size(), 128);
  ASSERT_EQ(cuda_tensor.nbytes(), 128 * sizeof(float));
  strides = cuda_tensor.strides();
  ASSERT_EQ(strides.size(), 2);
  ASSERT_EQ(strides[0], 4);
  ASSERT_EQ(strides[1], 1);
}

TEST(TensorTest, toDev) {
  auto cpu_allocator = ginfer::memory::CPUAllocatorFactory::get_instance();
  ASSERT_NE(cpu_allocator, nullptr);
  ginfer::tensor::Shape shape{16, 8};
  ginfer::tensor::Tensor tensor = ginfer::tensor::Tensor(
      ginfer::tensor::Dtype::kDtypeFloat32, shape, ginfer::common::DeviceType::kDeviceCPU);
  ASSERT_EQ(tensor.dtype(), ginfer::tensor::Dtype::kDtypeFloat32);
  ASSERT_EQ(tensor.shape().ndim(), 2);
  ASSERT_EQ(tensor.shape()[0], 16);
  ASSERT_EQ(tensor.shape()[1], 8);
  ASSERT_EQ(tensor.size(), 128);
  ASSERT_EQ(tensor.nbytes(), 128 * sizeof(float));

  // to cuda
  tensor.toDevice(ginfer::memory::DeviceType::kDeviceCUDA);
  // to cpu
  tensor.toDevice(ginfer::memory::DeviceType::kDeviceCPU);

  // ASSERT_EQ(cpu_tensor.shape)
}