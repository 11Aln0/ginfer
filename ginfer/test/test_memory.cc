#include <gtest/gtest.h>
#include <cstddef>

#include "ginfer/memory/allocator_factory.h"
#include "ginfer/memory/buffer.h"

namespace ginfer::test::memory {

using std::byte;

TEST(MemoryTest, DefaultCUDAAllocator) {
  auto allocator = ginfer::memory::DefaultGlobalCUDAAllocator::get_instance();
  ASSERT_NE(allocator, nullptr);
  void* ptr = allocator->alloc(1024);
  ASSERT_NE(ptr, nullptr);
  allocator->free(ptr, 1024);
}

TEST(MemoryTest, PooledCUDAAllocator) {
  using PooledGlobalCUDAAllocator = ginfer::memory::GlobalCUDAAllocator<ginfer::memory::cuda::PooledAllocStrategy>;
  auto allocator = PooledGlobalCUDAAllocator::get_instance();
  ASSERT_NE(allocator, nullptr);
  void* ptr1 = allocator->alloc(1024);
  ASSERT_NE(ptr1, nullptr);
  allocator->free(ptr1, 1024);

  void* ptr2 = allocator->alloc(2028);
  ASSERT_NE(ptr2, nullptr);
  ASSERT_EQ(ptr1, ptr2);  // should reuse the first block

  void* ptr3 = allocator->alloc(4096);
  ASSERT_NE(ptr3, nullptr);
  ASSERT_NE(ptr2, ptr3);  // should allocate a new block

  void* ptr4 = allocator->alloc(1024 * 1024);
  ASSERT_NE(ptr4, nullptr);

  void* ptr5 = allocator->alloc(2048 * 1024);
  ASSERT_NE(ptr5, nullptr);

  allocator->free(ptr2, 2028);
  allocator->free(ptr3, 4096);
  allocator->free(ptr4, 1024 * 1024);
  allocator->free(ptr5, 2048 * 1024);
}

TEST(MemoryTest, CPUAllocator) {
  auto allocator = ginfer::memory::GlobalCPUAllocator::get_instance();
  ASSERT_NE(allocator, nullptr);
  void* ptr = allocator->alloc(1024);
  ASSERT_NE((byte*)ptr, nullptr);
  allocator->free(ptr, 1024);
}

TEST(MemoryTest, CUDABuffer) {
  using ginfer::memory::Buffer;
  using ginfer::memory::DeviceType;
  Buffer buf(1024, DeviceType::kDeviceCUDA);
  ASSERT_EQ(buf.devType(), DeviceType::kDeviceCUDA);
  ASSERT_EQ(buf.size(), 1024);
  ASSERT_NE(buf.ptr(), nullptr);

  using PooledGlobalCUDAAllocator = ginfer::memory::GlobalCUDAAllocator<ginfer::memory::cuda::PooledAllocStrategy>;
  Buffer buf1 = Buffer(1024, PooledGlobalCUDAAllocator::get_instance());
  ASSERT_EQ(buf1.devType(), DeviceType::kDeviceCUDA);
  ASSERT_EQ(buf1.size(), 1024);
  ASSERT_NE(buf1.ptr(), nullptr);

  Buffer buf2 = Buffer(2048, ginfer::memory::DefaultGlobalCUDAAllocator::get_instance());
  ASSERT_EQ(buf2.devType(), DeviceType::kDeviceCUDA);
  ASSERT_EQ(buf2.size(), 2048);
  ASSERT_NE(buf2.ptr(), nullptr);
}

TEST(MemoryTest, CPUBuffer) {
  auto allocator = ginfer::memory::GlobalCPUAllocator::get_instance();
  ASSERT_NE(allocator, nullptr);
  {
    ginfer::memory::Buffer buffer(1024, allocator);
    ASSERT_EQ(buffer.devType(), ginfer::memory::DeviceType::kDeviceCPU);
    ASSERT_EQ(buffer.size(), 1024);
    ASSERT_NE(buffer.ptr(), nullptr);
  }

  {
    float* ptr = new float[32];
    ginfer::memory::Buffer ext_buffer(32 * sizeof(float), (byte*)ptr, ginfer::memory::DeviceType::kDeviceCPU);
    ASSERT_EQ(ext_buffer.devType(), ginfer::memory::DeviceType::kDeviceCPU);
    ASSERT_EQ(ext_buffer.size(), 32 * sizeof(float));
    ASSERT_EQ(ext_buffer.ptr(), (byte*)ptr);
    delete[] ptr;
  }
}

}  // namespace ginfer::test::memory