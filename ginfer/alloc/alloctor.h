#pragma once

#include <cstddef>
#include "device.h"
#include <memory>

enum class DeviceType {
    kDeviceUnknown,
    kDeviceCPU,
    kDeviceCUDA,
    kDeviceROCM
};

enum class MemcpyKind {
    kMemcpyHostToDevice,
    kMemcpyDeviceToHost
};

class DeviceAllocator {
public:
    explicit DeviceAllocator(DeviceType device_type): device_type(device_type) {};

    virtual void* alloc(size_t size) const = 0;

    virtual void free(void* ptr) const = 0;

    virtual void memcpy(const void* src, void* dst, size_t size, MemcpyKind kind) const;

    virtual void memset(void* ptr, size_t size, char c) ;

private:
    DeviceType device_type;
};

class CPUDeviceAllocator : public DeviceAllocator {
public:
    explicit CPUDeviceAllocator();

    void* alloc(size_t size) const override;

    void free(void* ptr) const override;

    void memcpy(const void* src, void* dst, size_t size, MemcpyKind kind) const override;
};

class CUDADeviceAllocator : public DeviceAllocator {
public:
    explicit CUDADeviceAllocator();

    void* alloc(size_t size) const override;

    void free(void* ptr) const override;

    void memcpy(const void* src, void* dst, size_t size, MemcpyKind kind) const override;
};  