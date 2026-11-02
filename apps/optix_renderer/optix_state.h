#pragma once
#include <optix.h>

// OptiX context, the single module/pipeline and the shader binding table.
struct OptixState {
    OptixDeviceContext context = nullptr;
    OptixModule module = nullptr;
    OptixProgramGroup raygen = nullptr, raygen_guides = nullptr, raygen_display = nullptr;
    OptixProgramGroup miss = nullptr, hitgroup = nullptr;
    OptixPipeline pipeline = nullptr;
    OptixShaderBindingTable sbt = {};           // path tracer (__raygen__path)
    OptixShaderBindingTable sbt_guides = {};    // denoiser guide pass (__raygen__guides)
    OptixShaderBindingTable sbt_display = {};   // viewer sRGB conversion (__raygen__display)
    CUdeviceptr sbt_buffer = 0;

    OptixState() = default;
    OptixState(const OptixState&) = delete;
    OptixState& operator=(const OptixState&) = delete;
    ~OptixState();

    void create_context(bool verbose);
    // max_registers = 0 lets OptiX choose.
    void create_pipeline(unsigned max_registers);
};
