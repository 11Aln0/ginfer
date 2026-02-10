#include <cuda_runtime_api.h>
#include <gtest/gtest.h>
#include <cstddef>

#include "ginfer/memory/allocator_factory.h"
#include "ginfer/tensor/tensor.h"

namespace ginfer::test::tensor {

using std::byte;

TEST(TensorTest, fromBuffer) {
  // cpu test

  auto cpu_allocator = ginfer::memory::GlobalCPUAllocator::getInstance();
  ASSERT_NE(cpu_allocator, nullptr);
  float* ptr1 = new float[128];
  auto cpu_buffer = std::make_shared<ginfer::memory::Buffer>(128 * sizeof(float), (byte*)ptr1,
                                                             ginfer::memory::DeviceType::kDeviceCPU);
  ginfer::tensor::Shape shape{32, 4};
  ginfer::tensor::Tensor cpu_tensor =
      ginfer::tensor::Tensor(ginfer::tensor::DataType::kDataTypeFloat32, shape, cpu_buffer);
  ASSERT_EQ(cpu_tensor.dtype(), ginfer::tensor::DataType::kDataTypeFloat32);
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

  auto cuda_allocator = ginfer::memory::DefaultGlobalCUDAAllocator::getInstance();
  ASSERT_NE(cuda_allocator, nullptr);
  float* ptr2 = nullptr;
  cudaError_t err = cudaMallocManaged(reinterpret_cast<void**>(&ptr2), 128 * sizeof(float));
  ASSERT_EQ(err, cudaSuccess);
  auto cuda_buffer = std::make_shared<ginfer::memory::Buffer>(128 * sizeof(float), (byte*)ptr2,
                                                              ginfer::memory::DeviceType::kDeviceCUDA);
  ginfer::tensor::Tensor cuda_tensor =
      ginfer::tensor::Tensor(ginfer::tensor::DataType::kDataTypeFloat32, shape, cuda_buffer);
  ASSERT_EQ(cuda_tensor.dtype(), ginfer::tensor::DataType::kDataTypeFloat32);
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

  auto cpu_allocator = ginfer::memory::GlobalCPUAllocator::getInstance();
  ASSERT_NE(cpu_allocator, nullptr);
  ginfer::tensor::Shape shape{32, 4};
  ginfer::tensor::Tensor cpu_tensor =
      ginfer::tensor::Tensor(ginfer::tensor::DataType::kDataTypeFloat32, shape, ginfer::common::DeviceType::kDeviceCPU);
  ASSERT_EQ(cpu_tensor.dtype(), ginfer::tensor::DataType::kDataTypeFloat32);
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

  auto cuda_allocator = ginfer::memory::DefaultGlobalCUDAAllocator::getInstance();
  ASSERT_NE(cuda_allocator, nullptr);
  ginfer::tensor::Tensor cuda_tensor = ginfer::tensor::Tensor(ginfer::tensor::DataType::kDataTypeFloat32, shape,
                                                              ginfer::common::DeviceType::kDeviceCUDA);
  ASSERT_EQ(cuda_tensor.dtype(), ginfer::tensor::DataType::kDataTypeFloat32);
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
  auto cpu_allocator = ginfer::memory::GlobalCPUAllocator::getInstance();
  ASSERT_NE(cpu_allocator, nullptr);
  ginfer::tensor::Shape shape{16, 8};
  ginfer::tensor::Tensor tensor =
      ginfer::tensor::Tensor(ginfer::tensor::DataType::kDataTypeFloat32, shape, ginfer::common::DeviceType::kDeviceCPU);
  ASSERT_EQ(tensor.dtype(), ginfer::tensor::DataType::kDataTypeFloat32);
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

TEST(TensorTest, strides) {
  auto cpu_allocator = ginfer::memory::GlobalCPUAllocator::getInstance();
  ASSERT_NE(cpu_allocator, nullptr);
  ginfer::tensor::Shape shape{4, 3, 2};
  ginfer::tensor::Tensor row_major_tensor =
      ginfer::tensor::Tensor(ginfer::tensor::DataType::kDataTypeFloat32, shape, ginfer::common::DeviceType::kDeviceCPU);
  auto row_strides = row_major_tensor.strides();
  ASSERT_EQ(row_strides.size(), 3);
  ASSERT_EQ(row_strides[0], 6);
  ASSERT_EQ(row_strides[1], 2);
  ASSERT_EQ(row_strides[2], 1);

  ginfer::tensor::Tensor col_major_tensor =
      ginfer::tensor::Tensor(ginfer::tensor::DataType::kDataTypeFloat32, shape, ginfer::common::DeviceType::kDeviceCPU);
  col_major_tensor.permute({2, 1, 0});
  auto col_strides = col_major_tensor.strides();
  ASSERT_EQ(col_strides.size(), 3);
  ASSERT_EQ(col_strides[0], 1);
  ASSERT_EQ(col_strides[1], 4);
  ASSERT_EQ(col_strides[2], 12);
}

TEST(TensorTest, slice) {
  ginfer::tensor::Shape shape{4, 3, 2};
  ginfer::tensor::Tensor tensor(ginfer::tensor::DataType::kDataTypeFloat32, shape,
                                ginfer::common::DeviceType::kDeviceCPU);

  // Fill with known values: 0, 1, 2, ..., 23
  float* ptr = tensor.data<float>();
  for (int i = 0; i < 24; i++) {
    ptr[i] = static_cast<float>(i);
  }

  // Slice dim 0: [1, 3) -> shape {2, 3, 2}
  auto sliced0 = tensor.slice(0, 1, 3);
  ASSERT_EQ(sliced0->shape().ndim(), 3);
  ASSERT_EQ(sliced0->shape()[0], 2);
  ASSERT_EQ(sliced0->shape()[1], 3);
  ASSERT_EQ(sliced0->shape()[2], 2);
  ASSERT_EQ(sliced0->size(), 12);

  // Strides should be preserved from the original tensor
  auto orig_strides = tensor.strides();
  auto s0_strides = sliced0->strides();
  ASSERT_EQ(s0_strides.size(), 3);
  ASSERT_EQ(s0_strides[0], orig_strides[0]);
  ASSERT_EQ(s0_strides[1], orig_strides[1]);
  ASSERT_EQ(s0_strides[2], orig_strides[2]);

  // Data should point to element [1,0,0] = index 1*6 = 6
  float* s0_ptr = sliced0->data<float>();
  ASSERT_EQ(s0_ptr[0], 6.0f);

  // Slice dim 1: [0, 2) -> shape {4, 2, 2}
  auto sliced1 = tensor.slice(1, 0, 2);
  ASSERT_EQ(sliced1->shape()[0], 4);
  ASSERT_EQ(sliced1->shape()[1], 2);
  ASSERT_EQ(sliced1->shape()[2], 2);
  ASSERT_EQ(sliced1->size(), 16);

  // Slice dim 2: [1, 2) -> shape {4, 3, 1}
  auto sliced2 = tensor.slice(2, 1, 2);
  ASSERT_EQ(sliced2->shape()[0], 4);
  ASSERT_EQ(sliced2->shape()[1], 3);
  ASSERT_EQ(sliced2->shape()[2], 1);
  ASSERT_EQ(sliced2->size(), 12);

  // Data should point to element [0,0,1] = index 1
  float* s2_ptr = sliced2->data<float>();
  ASSERT_EQ(s2_ptr[0], 1.0f);

  // Chained slice: slice dim 0 [1,3), then slice dim 1 [1,2) on the result
  auto chained = sliced0->slice(1, 1, 2);
  ASSERT_EQ(chained->shape()[0], 2);
  ASSERT_EQ(chained->shape()[1], 1);
  ASSERT_EQ(chained->shape()[2], 2);
  // Should point to element [1,1,0] = index 1*6 + 1*2 = 8
  float* ch_ptr = chained->data<float>();
  ASSERT_EQ(ch_ptr[0], 8.0f);
}

TEST(TensorTest, reshape) {
  ginfer::tensor::Shape shape{4, 3, 2};
  ginfer::tensor::Tensor tensor(ginfer::tensor::DataType::kDataTypeFloat32, shape,
                                ginfer::common::DeviceType::kDeviceCPU);

  float* ptr = tensor.data<float>();
  for (int i = 0; i < 24; i++) {
    ptr[i] = static_cast<float>(i);
  }

  // Reshape to {6, 4}
  auto reshaped = tensor.reshape(ginfer::tensor::Shape{6, 4});
  ASSERT_EQ(reshaped->shape().ndim(), 2);
  ASSERT_EQ(reshaped->shape()[0], 6);
  ASSERT_EQ(reshaped->shape()[1], 4);
  ASSERT_EQ(reshaped->size(), 24);

  // Strides should be recalculated for the new shape
  auto r_strides = reshaped->strides();
  ASSERT_EQ(r_strides.size(), 2);
  ASSERT_EQ(r_strides[0], 4);
  ASSERT_EQ(r_strides[1], 1);

  // Data should be identical (shared buffer)
  float* r_ptr = reshaped->data<float>();
  for (int i = 0; i < 24; i++) {
    ASSERT_EQ(r_ptr[i], static_cast<float>(i));
  }

  // Reshape to flat {24}
  auto flat = tensor.reshape(ginfer::tensor::Shape{24});
  ASSERT_EQ(flat->shape().ndim(), 1);
  ASSERT_EQ(flat->shape()[0], 24);
  ASSERT_EQ(flat->size(), 24);
  auto flat_strides = flat->strides();
  ASSERT_EQ(flat_strides.size(), 1);
  ASSERT_EQ(flat_strides[0], 1);

  // Reshape to higher rank {2, 3, 2, 2}
  auto higher = tensor.reshape(ginfer::tensor::Shape{2, 3, 2, 2});
  ASSERT_EQ(higher->shape().ndim(), 4);
  ASSERT_EQ(higher->shape()[0], 2);
  ASSERT_EQ(higher->shape()[1], 3);
  ASSERT_EQ(higher->shape()[2], 2);
  ASSERT_EQ(higher->shape()[3], 2);
  ASSERT_EQ(higher->size(), 24);

  // Reshape with mismatched numel should fail
  // ASSERT_THROW(tensor.reshape(ginfer::tensor::Shape{5, 5}), std::exception);
}

}  // namespace ginfer::test::tensor