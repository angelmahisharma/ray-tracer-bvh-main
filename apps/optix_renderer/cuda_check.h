#pragma once
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <optix.h>

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t e_ = (call);                                                  \
        if (e_ != cudaSuccess) {                                                  \
            std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__, \
                         __LINE__, cudaGetErrorString(e_));                       \
            std::exit(1);                                                         \
        }                                                                         \
    } while (0)

#define OPTIX_CHECK(call)                                                          \
    do {                                                                           \
        OptixResult r_ = (call);                                                   \
        if (r_ != OPTIX_SUCCESS) {                                                 \
            std::fprintf(stderr, "OptiX error %s at %s:%d: %s\n", #call, __FILE__, \
                         __LINE__, optixGetErrorString(r_));                       \
            std::exit(1);                                                          \
        }                                                                          \
    } while (0)

// Typed device buffer: upload a vector, free on destruction.
template <typename T>
struct DeviceBuffer {
    T* ptr = nullptr;
    size_t count = 0;

    DeviceBuffer() = default;
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    ~DeviceBuffer() { if (ptr) cudaFree(ptr); }

    void alloc(size_t n) {
        if (ptr) cudaFree(ptr);
        ptr = nullptr;
        count = n;
        if (n) CUDA_CHECK(cudaMalloc(&ptr, n * sizeof(T)));
    }
    template <typename Vec>
    void upload(const Vec& v) {
        alloc(v.size());
        if (count) CUDA_CHECK(cudaMemcpy(ptr, v.data(), count * sizeof(T), cudaMemcpyHostToDevice));
    }
    CUdeviceptr dptr() const { return (CUdeviceptr)ptr; }
};
