#include <gtest/gtest.h>

#include "ginfer/memory/allocator.h"
#include "ginfer/memory/buffer.h"

TEST(MemoryTest, CUDAAllocator) {
  auto allocator = ginfer::memory::CUDAAllocatorFactory::get_instance();
  ASSERT_NE(allocator, nullptr);
  void* ptr = allocator->alloc(1024);
  ASSERT_NE(ptr, nullptr);
  allocator->free(ptr);
}

TEST(MemoryTest, CPUAllocator) {
  auto allocator = ginfer::memory::CPUAllocatorFactory::get_instance();
  ASSERT_NE(allocator, nullptr);
  void* ptr = allocator->alloc(1024);
  ASSERT_NE(ptr, nullptr);
  allocator->free(ptr);
}

TEST(MemoryTest, CUDABuffer) {
  // auto allocator = ginfer::memory::CUDAAllocatorFactory::get_instance();
  // ASSERT_NE(allocator, nullptr);
  ginfer::memory::Buffer buffer(1024, ginfer::memory::DeviceType::kDeviceCUDA);
  ASSERT_EQ(buffer.devType(), ginfer::memory::DeviceType::kDeviceCUDA);
  ASSERT_EQ(buffer.size(), 1024);
  ASSERT_NE(buffer.ptr(), nullptr);
}

TEST(MemoryTest, CPUBuffer) {
  auto allocator = ginfer::memory::CPUAllocatorFactory::get_instance();
  ASSERT_NE(allocator, nullptr);
  {
    ginfer::memory::Buffer buffer(1024, ginfer::memory::DeviceType::kDeviceCPU);
    ASSERT_EQ(buffer.devType(), ginfer::memory::DeviceType::kDeviceCPU);
    ASSERT_EQ(buffer.size(), 1024);
    ASSERT_NE(buffer.ptr(), nullptr);
  }

  {
    float* ptr = new float[32];
    ginfer::memory::Buffer ext_buffer(32 * sizeof(float), ptr,
                                      ginfer::memory::DeviceType::kDeviceCPU);
    ASSERT_EQ(ext_buffer.devType(), ginfer::memory::DeviceType::kDeviceCPU);
    ASSERT_EQ(ext_buffer.size(), 32 * sizeof(float));
    ASSERT_EQ(ext_buffer.ptr(), ptr);
    delete[] ptr;
  }
}