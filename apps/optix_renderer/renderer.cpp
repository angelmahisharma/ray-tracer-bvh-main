#include "renderer.h"
#include <stdexcept>
#include <optix_stubs.h>
#include "device_scene.h"

using namespace rt;

static float3 f3(const Vec3& v) { return float3{v.x, v.y, v.z}; }

Renderer::Renderer(const Scene& scene, const RenderOptions& options) : options_(options) {
    if (options_.upscale2x) options_.denoise = true;
    out_width_ = (unsigned)scene.settings.width;
    out_height_ = (unsigned)scene.settings.height;
    if (options_.upscale2x && (out_width_ % 2 || out_height_ % 2))
        throw std::runtime_error("upscale2x needs an even image width and height");
    width_ = options_.upscale2x ? out_width_ / 2 : out_width_;
    height_ = options_.upscale2x ? out_height_ / 2 : out_height_;

    optix_.create_context(options_.verbose);
    optix_.create_pipeline(options_.max_registers);

    const DeviceSceneData data = pack_scene(scene);
    build_gas(optix_.context, scene, data.tri_sbt_offset, gas_);
    tris_.upload(data.tris);
    materials_.upload(data.materials);
    lights_.upload(data.lights);
    light_cdf_.upload(data.light_cdf);
    tri_to_light_.upload(data.tri_to_light);
    accum_.alloc((size_t)width_ * height_);
    if (options_.denoise) {
        albedo_.alloc((size_t)width_ * height_);
        normal_.alloc((size_t)width_ * height_);
        denoised_.alloc((size_t)out_width_ * out_height_);
    }

    params_.handle = gas_.handle;
    params_.accum = accum_.ptr;
    params_.tris = tris_.ptr;
    params_.materials = materials_.ptr;
    params_.lights = lights_.ptr;
    params_.light_cdf = light_cdf_.ptr;
    params_.tri_to_light = tri_to_light_.ptr;
    params_.albedo = albedo_.ptr;
    params_.normal = normal_.ptr;
    params_.env_mode = scene.env.mode;
    params_.env_a = f3(scene.env.a);
    params_.env_b = f3(scene.env.b);
    params_.width = width_;
    params_.height = height_;
    params_.spp = (uint32_t)scene.settings.spp;
    params_.max_bounces = (uint32_t)scene.settings.max_bounces;
    params_.num_lights = scene.total_light_power > 0.f ? (uint32_t)data.lights.size() : 0u;
    params_.seed = options_.seed;
    params_.reorder = options_.reorder ? 1u : 0u;
    params_.guide_spp = options_.guide_spp;
    set_camera(scene.camera);
    d_params_.alloc(1);

    CUDA_CHECK(cudaStreamCreate(&stream_));
    if (options_.denoise) denoiser_.create(optix_.context, stream_, width_, height_, options_.upscale2x);
}

Renderer::~Renderer() {
    if (stream_) {
        cudaStreamSynchronize(stream_);
        cudaStreamDestroy(stream_);
    }
}

void Renderer::set_camera(const Camera& camera) {
    params_.cam_origin = f3(camera.origin);
    params_.cam_lower_left = f3(camera.lower_left);
    params_.cam_horizontal = f3(camera.horizontal);
    params_.cam_vertical = f3(camera.vertical);
    params_dirty_ = true;
}

void Renderer::set_spp(uint32_t spp) {
    params_.spp = spp > 0 ? spp : 1;
    params_dirty_ = true;
}

void Renderer::set_frame(uint32_t frame_index, uint32_t seed) {
    params_.frame_index = frame_index;
    params_.seed = seed;
    params_dirty_ = true;
}

void Renderer::launch(const OptixShaderBindingTable& sbt, unsigned w, unsigned h) {
    if (params_dirty_) {
        CUDA_CHECK(cudaMemcpyAsync(d_params_.ptr, &params_, sizeof(params_), cudaMemcpyHostToDevice, stream_));
        params_dirty_ = false;
    }
    OPTIX_CHECK(optixLaunch(optix_.pipeline, stream_, d_params_.dptr(), sizeof(LaunchParams), &sbt, w, h, 1));
}

void Renderer::trace() { launch(optix_.sbt, width_, height_); }

void Renderer::trace_guides() { launch(optix_.sbt_guides, width_, height_); }

void Renderer::denoise() { denoiser_.denoise(stream_, accum_.ptr, albedo_.ptr, normal_.ptr, denoised_.ptr); }

void Renderer::to_display(const float4* src, uchar4* dst) {
    if (params_.display_src != src || params_.display != dst) {
        params_.display_src = src;
        params_.display = dst;
        params_dirty_ = true;
    }
    launch(optix_.sbt_display, out_width_, out_height_);
}

unsigned long long Renderer::count_rays() {
    DeviceBuffer<unsigned long long> counter;
    counter.alloc(1);
    CUDA_CHECK(cudaMemset(counter.ptr, 0, sizeof(unsigned long long)));
    params_.ray_counter = counter.ptr;
    params_dirty_ = true;
    trace();
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    unsigned long long count = 0;
    CUDA_CHECK(cudaMemcpy(&count, counter.ptr, sizeof(count), cudaMemcpyDeviceToHost));
    params_.ray_counter = nullptr;
    params_dirty_ = true;
    return count;
}
