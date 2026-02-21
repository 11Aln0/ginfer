#include <gtest/gtest.h>
#include <cstddef>

#include "ginfer/memory/alloc_strategy.h"
#include "ginfer/memory/allocator_factory.h"
#include "ginfer/memory/buffer.h"

namespace ginfer::test::memory {

using std::byte;

TEST(MemoryTest, DefaultCUDAAllocator) {
  auto allocator = ginfer::memory::DefaultGlobalCUDAAllocator::getInstance();
  ASSERT_NE(allocator, nullptr);
  auto res = allocator->alloc(1024);
  ASSERT_TRUE(res.ok()) << "Allocation failed with error: " << res.err();
  void* ptr = res.value();
  ASSERT_NE(ptr, nullptr);
  allocator->free(ptr, 1024);
}

TEST(MemoryTest, PooledCUDAAllocator) {
  using namespace ginfer::memory;
  using PooledCUDADAllocator = PooledAllocStrategy<CUDADeviceAllocator>;
  using PooledGlobalCUDAAllocator = GlobalDeviceAllocator<PooledCUDADAllocator>;
  auto allocator = PooledGlobalCUDAAllocator::getInstance();
  ASSERT_NE(allocator, nullptr);

  auto res1 = allocator->alloc(1024);
  ASSERT_TRUE(res1.ok()) << res1.err();
  void* ptr1 = res1.value();
  ASSERT_NE(ptr1, nullptr);
  allocator->free(ptr1, 1024);

  auto res2 = allocator->alloc(2028);
  ASSERT_TRUE(res2.ok()) << res2.err();
  void* ptr2 = res2.value();
  ASSERT_NE(ptr2, nullptr);
  ASSERT_EQ(ptr1, ptr2);  // should reuse the first block

  auto res3 = allocator->alloc(4096);
  ASSERT_TRUE(res3.ok()) << res3.err();
  void* ptr3 = res3.value();
  ASSERT_NE(ptr3, nullptr);
  ASSERT_NE(ptr2, ptr3);  // should allocate a new block

  auto res4 = allocator->alloc(1024 * 1024);
  ASSERT_TRUE(res4.ok()) << res4.err();
  void* ptr4 = res4.value();
  ASSERT_NE(ptr4, nullptr);

  auto res5 = allocator->alloc(2048 * 1024);
  ASSERT_TRUE(res5.ok()) << res5.err();
  void* ptr5 = res5.value();
  ASSERT_NE(ptr5, nullptr);

  allocator->free(ptr2, 2028);
  allocator->free(ptr3, 4096);
  allocator->free(ptr4, 1024 * 1024);
  allocator->free(ptr5, 2048 * 1024);
}

TEST(MemoryTest, CPUAllocator) {
  auto allocator = ginfer::memory::GlobalCPUAllocator::getInstance();
  ASSERT_NE(allocator, nullptr);
  auto res = allocator->alloc(1024);
  ASSERT_TRUE(res.ok()) << "Allocation failed with error: " << res.err();
  void* ptr = res.value();
  ASSERT_NE((byte*)ptr, nullptr);
  allocator->free(ptr, 1024);
}

TEST(MemoryTest, CUDABuffer) {
  using ginfer::memory::Buffer;
  using ginfer::memory::DeviceType;
  auto res = Buffer::create(1024, DeviceType::kDeviceCUDA);
  ASSERT_TRUE(res.ok()) << "Buffer creation failed with error: " << res.err();
  std::shared_ptr<Buffer> buf = res.value();
  ASSERT_EQ(buf->devType(), DeviceType::kDeviceCUDA);
  ASSERT_EQ(buf->size(), 1024);
  ASSERT_NE(buf->ptr(), nullptr);

  using PooledGlobalCUDAAllocator = ginfer::memory::GlobalDeviceAllocator<ginfer::memory::PooledAllocStrategy<ginfer::memory::CUDADeviceAllocator>>;
  auto res1 = Buffer::create(1024, PooledGlobalCUDAAllocator::getInstance());
  ASSERT_TRUE(res1.ok()) << res1.err();
  std::shared_ptr<Buffer> buf1 = res1.value();
  ASSERT_EQ(buf1->devType(), DeviceType::kDeviceCUDA);
  ASSERT_EQ(buf1->size(), 1024);
  ASSERT_NE(buf1->ptr(), nullptr);

  auto res2 = Buffer::create(2048, ginfer::memory::DefaultGlobalCUDAAllocator::getInstance());
  ASSERT_TRUE(res2.ok()) << res2.err();
  std::shared_ptr<Buffer> buf2 = res2.value();
  ASSERT_EQ(buf2->devType(), DeviceType::kDeviceCUDA);
  ASSERT_EQ(buf2->size(), 2048);
  ASSERT_NE(buf2->ptr(), nullptr);
}

TEST(MemoryTest, CPUBuffer) {
  auto allocator = ginfer::memory::GlobalCPUAllocator::getInstance();
  ASSERT_NE(allocator, nullptr);
  {
    auto res = ginfer::memory::Buffer::create(1024, allocator);
    ASSERT_TRUE(res.ok()) << res.err();
    std::shared_ptr<ginfer::memory::Buffer> buffer = res.value();
    ASSERT_EQ(buffer->devType(), ginfer::memory::DeviceType::kDeviceCPU);
    ASSERT_EQ(buffer->size(), 1024);
    ASSERT_NE(buffer->ptr(), nullptr);
  }

  {
    float* ptr = new float[32];
    auto res = ginfer::memory::Buffer::create(32 * sizeof(float), (byte*)ptr, ginfer::memory::DeviceType::kDeviceCPU);
    ASSERT_TRUE(res.ok()) << res.err();
    std::shared_ptr<ginfer::memory::Buffer> ext_buffer = res.value();
    ASSERT_EQ(ext_buffer->devType(), ginfer::memory::DeviceType::kDeviceCPU);
    ASSERT_EQ(ext_buffer->size(), 32 * sizeof(float));
    ASSERT_EQ(ext_buffer->ptr(), (byte*)ptr);
    delete[] ptr;
  }
}

}  // namespace ginfer::test::memory