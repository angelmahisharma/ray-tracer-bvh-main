#include "gas.h"
#include <chrono>
#include <optix_stubs.h>
#include "cuda_check.h"
#include "launch_params.h"

Gas::~Gas() {
    if (buffer) cudaFree((void*)buffer);
}

void build_gas(OptixDeviceContext ctx, const rt::Scene& scene,
               const std::vector<uint8_t>& sbt_offsets, Gas& out) {
    // Positions only: 3 vertices per triangle, no index buffer (DATA_LAYOUT.md).
    // The GAS keeps its own copy, so these buffers are freed after the build.
    DeviceBuffer<float> d_pos;
    d_pos.upload(scene.flat_positions());
    DeviceBuffer<uint8_t> d_offsets;
    d_offsets.upload(sbt_offsets);

    // Everything is opaque: the RT cores never call back into an any-hit program.
    uint32_t flags[kNumHitRecords];
    for (uint32_t& f : flags) f = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;

    CUdeviceptr vertex_buffer = d_pos.dptr();
    OptixBuildInput input = {};
    input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    OptixBuildInputTriangleArray& tris = input.triangleArray;
    tris.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    tris.vertexStrideInBytes = 3 * sizeof(float);
    tris.numVertices = (unsigned)(d_pos.count / 3);
    tris.vertexBuffers = &vertex_buffer;
    tris.flags = flags;
    tris.numSbtRecords = kNumHitRecords;
    tris.sbtIndexOffsetBuffer = d_offsets.dptr();
    tris.sbtIndexOffsetSizeInBytes = sizeof(uint8_t);
    tris.sbtIndexOffsetStrideInBytes = sizeof(uint8_t);

    OptixAccelBuildOptions options = {};
    options.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE | OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    options.operation = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes sizes = {};
    OPTIX_CHECK(optixAccelComputeMemoryUsage(ctx, &options, &input, 1, &sizes));

    DeviceBuffer<char> temp;
    temp.alloc(sizes.tempSizeInBytes);
    CUdeviceptr uncompacted = 0;
    CUDA_CHECK(cudaMalloc((void**)&uncompacted, sizes.outputSizeInBytes));
    DeviceBuffer<uint64_t> d_compacted_size;
    d_compacted_size.alloc(1);
    OptixAccelEmitDesc emit = {d_compacted_size.dptr(), OPTIX_PROPERTY_TYPE_COMPACTED_SIZE};

    CUDA_CHECK(cudaDeviceSynchronize());
    const auto t0 = std::chrono::steady_clock::now();

    OPTIX_CHECK(optixAccelBuild(ctx, 0, &options, &input, 1, temp.dptr(), sizes.tempSizeInBytes,
                                uncompacted, sizes.outputSizeInBytes, &out.handle, &emit, 1));
    uint64_t compacted_size = 0;
    CUDA_CHECK(cudaMemcpy(&compacted_size, d_compacted_size.ptr, sizeof(uint64_t), cudaMemcpyDeviceToHost));

    out.uncompacted_bytes = sizes.outputSizeInBytes;
    if (compacted_size < sizes.outputSizeInBytes) {
        CUDA_CHECK(cudaMalloc((void**)&out.buffer, compacted_size));
        OPTIX_CHECK(optixAccelCompact(ctx, 0, out.handle, out.buffer, compacted_size, &out.handle));
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaFree((void*)uncompacted));
        out.bytes = compacted_size;
    } else {
        out.buffer = uncompacted;
        out.bytes = sizes.outputSizeInBytes;
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    out.build_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}
