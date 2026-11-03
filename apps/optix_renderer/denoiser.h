#pragma once
#include <cuda_runtime.h>
#include <optix.h>

struct Denoiser {
    OptixDenoiser handle = nullptr;
    CUdeviceptr state = 0, scratch = 0;
    size_t state_bytes = 0, scratch_bytes = 0;
    unsigned in_width = 0, in_height = 0, out_width = 0, out_height = 0;

    Denoiser() = default;
    Denoiser(const Denoiser&) = delete;
    Denoiser& operator=(const Denoiser&) = delete;
    ~Denoiser();

    void create(OptixDeviceContext ctx, cudaStream_t stream, unsigned w, unsigned h, bool upscale2x);
    // color/albedo/normal: in_width*in_height, output: out_width*out_height float4
    // device buffers. Asynchronous on `stream`.
    void denoise(cudaStream_t stream, const float4* color, const float4* albedo, const float4* normal,
                 float4* output);
};
