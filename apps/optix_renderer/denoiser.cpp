#include "denoiser.h"
#include <optix_stubs.h>
#include "cuda_check.h"

static OptixImage2D float4_image(const float4* data, unsigned w, unsigned h) {
    OptixImage2D img = {};
    img.data = (CUdeviceptr)data;
    img.width = w;
    img.height = h;
    img.rowStrideInBytes = w * (unsigned)sizeof(float4);
    img.pixelStrideInBytes = (unsigned)sizeof(float4);
    img.format = OPTIX_PIXEL_FORMAT_FLOAT4;
    return img;
}

void Denoiser::create(OptixDeviceContext ctx, cudaStream_t stream, unsigned w, unsigned h, bool upscale2x) {
    in_width = w;
    in_height = h;
    out_width = upscale2x ? 2 * w : w;
    out_height = upscale2x ? 2 * h : h;
    OptixDenoiserOptions options = {};
    options.guideAlbedo = 1;
    options.guideNormal = 1;
    options.denoiseAlpha = OPTIX_DENOISER_ALPHA_MODE_COPY;
    const OptixDenoiserModelKind kind = upscale2x ? OPTIX_DENOISER_MODEL_KIND_UPSCALE2X : OPTIX_DENOISER_MODEL_KIND_AOV;
    OPTIX_CHECK(optixDenoiserCreate(ctx, kind, &options, &handle));

    OptixDenoiserSizes sizes = {};
    OPTIX_CHECK(optixDenoiserComputeMemoryResources(handle, w, h, &sizes));
    state_bytes = sizes.stateSizeInBytes;
    scratch_bytes = sizes.withoutOverlapScratchSizeInBytes;   // whole image, no tiling
    CUDA_CHECK(cudaMalloc((void**)&state, state_bytes));
    CUDA_CHECK(cudaMalloc((void**)&scratch, scratch_bytes));
    OPTIX_CHECK(optixDenoiserSetup(handle, stream, w, h, state, state_bytes, scratch, scratch_bytes));
}

void Denoiser::denoise(cudaStream_t stream, const float4* color, const float4* albedo, const float4* normal,
                       float4* output) {
    OptixDenoiserGuideLayer guides = {};
    guides.albedo = float4_image(albedo, in_width, in_height);
    guides.normal = float4_image(normal, in_width, in_height);

    OptixDenoiserLayer layer = {};
    layer.input = float4_image(color, in_width, in_height);
    layer.output = float4_image(output, out_width, out_height);

    // hdrAverageColor left null: the denoiser computes the image's average log colour itself.
    OptixDenoiserParams params = {};
    OPTIX_CHECK(optixDenoiserInvoke(handle, stream, &params, state, state_bytes, &guides, &layer, 1, 0, 0, scratch,
                                    scratch_bytes));
}

Denoiser::~Denoiser() {
    if (scratch) cudaFree((void*)scratch);
    if (state) cudaFree((void*)state);
    if (handle) optixDenoiserDestroy(handle);
}
