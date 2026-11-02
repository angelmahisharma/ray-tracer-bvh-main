#pragma once
#include <cstdint>
#include "cuda_check.h"
#include "denoiser.h"
#include "gas.h"
#include "launch_params.h"
#include "optix_state.h"
#include "rt/scene.h"

struct RenderOptions {
    unsigned max_registers = 0;   // 0 = OptiX default
    bool reorder = true;          // Shader Execution Reordering
    bool verbose = false;         // OptiX log level 4
    bool denoise = false;         // allocate guide buffers + denoiser
    bool upscale2x = false;       // path-trace at half size, denoiser upscales (implies denoise)
    uint32_t guide_spp = 4;
    uint32_t seed = 1234;         // same global seed as cpu_renderer
};

class Renderer {
public:
    // Throws std::runtime_error on invalid options (e.g. upscale2x with an odd image size).
    Renderer(const rt::Scene& scene, const RenderOptions& options);
    ~Renderer();
    Renderer(const Renderer&) = delete;
    Renderer& operator=(const Renderer&) = delete;

    // Parameter changes are uploaded by the next launch.
    void set_camera(const rt::Camera& camera);
    void set_spp(uint32_t spp);
    // frame_index > 0 averages the next trace() into the running mean in accum().
    void set_frame(uint32_t frame_index, uint32_t seed);

    // Asynchronous on stream().
    void trace();          // path tracing -> accum()
    void trace_guides();   // albedo/normal guides (needs denoising enabled)
    void denoise();        // accum() + guides -> denoised() (needs denoising enabled)
    // src (out_width x out_height) -> 8-bit sRGB in dst, bottom row first (OpenGL).
    void to_display(const float4* src, uchar4* dst);
    // Synchronous: one extra trace() that counts rays.
    unsigned long long count_rays();

    unsigned width() const { return width_; }           // path-traced size
    unsigned height() const { return height_; }
    unsigned out_width() const { return out_width_; }   // final image size
    unsigned out_height() const { return out_height_; }
    bool denoising() const { return options_.denoise; }
    uint32_t spp() const { return params_.spp; }
    cudaStream_t stream() const { return stream_; }
    const Gas& gas() const { return gas_; }
    const float4* accum() const { return accum_.ptr; }
    const float4* albedo() const { return albedo_.ptr; }
    const float4* normal() const { return normal_.ptr; }
    const float4* denoised() const { return denoised_.ptr; }

private:
    void launch(const OptixShaderBindingTable& sbt, unsigned w, unsigned h);

    RenderOptions options_;
    unsigned width_ = 0, height_ = 0, out_width_ = 0, out_height_ = 0;
    OptixState optix_;
    Gas gas_;
    DeviceBuffer<TriShade> tris_;
    DeviceBuffer<DevMaterial> materials_;
    DeviceBuffer<DevLight> lights_;
    DeviceBuffer<float> light_cdf_;
    DeviceBuffer<int32_t> tri_to_light_;
    DeviceBuffer<float4> accum_, albedo_, normal_, denoised_;
    DeviceBuffer<LaunchParams> d_params_;
    LaunchParams params_ = {};
    bool params_dirty_ = true;
    cudaStream_t stream_ = nullptr;
    Denoiser denoiser_;
};
