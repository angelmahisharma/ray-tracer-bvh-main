#include "optix_state.h"
#include <cstdio>
#include <vector>
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>
#include "cuda_check.h"
#include "launch_params.h"
#include "programs_optixir.h"   // generated: kProgramsOptixIR

namespace {

void log_callback(unsigned level, const char* tag, const char* message, void*) {
    std::fprintf(stderr, "[optix %u][%s] %s\n", level, tag, message);
}

// Every SBT record is header-only: all data comes from LaunchParams.
struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) EmptyRecord {
    char header[OPTIX_SBT_RECORD_HEADER_SIZE];
};

}  // namespace

void OptixState::create_context(bool verbose) {
    CUDA_CHECK(cudaFree(nullptr));   // make sure the CUDA context exists
    OPTIX_CHECK(optixInit());
    OptixDeviceContextOptions options = {};
    options.logCallbackFunction = log_callback;
    options.logCallbackLevel = verbose ? 4 : 2;
    OPTIX_CHECK(optixDeviceContextCreate(nullptr, &options, &context));
}

void OptixState::create_pipeline(unsigned max_registers) {
    OptixModuleCompileOptions module_options = {};
    module_options.maxRegisterCount = max_registers;
    module_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    module_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;   // line info only, no cost

    // One GAS, triangles only, no payload (rays use optixTraverse + hit objects),
    // two attributes for the built-in barycentrics.
    OptixPipelineCompileOptions pipeline_options = {};
    pipeline_options.usesMotionBlur = 0;
    pipeline_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pipeline_options.numPayloadValues = 0;
    pipeline_options.numAttributeValues = 2;
    pipeline_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    pipeline_options.pipelineLaunchParamsVariableName = "params";
    pipeline_options.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;

    char log[4096];
    size_t log_size = sizeof(log);
    OPTIX_CHECK(optixModuleCreate(context, &module_options, &pipeline_options,
                                  reinterpret_cast<const char*>(kProgramsOptixIR), sizeof(kProgramsOptixIR),
                                  log, &log_size, &module));

    // Raygen does all the work. Miss and hit groups are empty: OptiX still needs
    // the records, but nothing is ever invoked (rays are never optixInvoke'd).
    OptixProgramGroupOptions group_options = {};
    OptixProgramGroupDesc rg = {};
    rg.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    rg.raygen.module = module;
    rg.raygen.entryFunctionName = "__raygen__path";
    OptixProgramGroupDesc rg_guides = rg;
    rg_guides.raygen.entryFunctionName = "__raygen__guides";
    OptixProgramGroupDesc rg_display = rg;
    rg_display.raygen.entryFunctionName = "__raygen__display";
    OptixProgramGroupDesc ms = {};
    ms.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    OptixProgramGroupDesc hg = {};
    hg.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;

    log_size = sizeof(log);
    OPTIX_CHECK(optixProgramGroupCreate(context, &rg, 1, &group_options, log, &log_size, &raygen));
    log_size = sizeof(log);
    OPTIX_CHECK(optixProgramGroupCreate(context, &rg_guides, 1, &group_options, log, &log_size, &raygen_guides));
    log_size = sizeof(log);
    OPTIX_CHECK(optixProgramGroupCreate(context, &rg_display, 1, &group_options, log, &log_size, &raygen_display));
    log_size = sizeof(log);
    OPTIX_CHECK(optixProgramGroupCreate(context, &ms, 1, &group_options, log, &log_size, &miss));
    log_size = sizeof(log);
    OPTIX_CHECK(optixProgramGroupCreate(context, &hg, 1, &group_options, log, &log_size, &hitgroup));

    OptixProgramGroup groups[] = {raygen, raygen_guides, raygen_display, miss, hitgroup};
    OptixPipelineLinkOptions link_options = {};
    link_options.maxTraceDepth = 1;   // rays are only cast from raygen
    log_size = sizeof(log);
    OPTIX_CHECK(optixPipelineCreate(context, &pipeline_options, &link_options, groups, 5, log, &log_size,
                                    &pipeline));

    OptixStackSizes stack_sizes = {};
    for (OptixProgramGroup g : groups) OPTIX_CHECK(optixUtilAccumulateStackSizes(g, &stack_sizes, pipeline));
    unsigned dc_from_traversal = 0, dc_from_state = 0, continuation = 0;
    OPTIX_CHECK(optixUtilComputeStackSizes(&stack_sizes, 1, 0, 0, &dc_from_traversal, &dc_from_state,
                                           &continuation));
    OPTIX_CHECK(optixPipelineSetStackSize(pipeline, dc_from_traversal, dc_from_state, continuation, 1));

    // SBT: [raygen path][raygen guides][raygen display][miss][hit x kNumHitRecords].
    // Hit record index = material type, which is what optixReorder() sorts on.
    std::vector<EmptyRecord> records(4 + kNumHitRecords);
    OPTIX_CHECK(optixSbtRecordPackHeader(raygen, &records[0]));
    OPTIX_CHECK(optixSbtRecordPackHeader(raygen_guides, &records[1]));
    OPTIX_CHECK(optixSbtRecordPackHeader(raygen_display, &records[2]));
    OPTIX_CHECK(optixSbtRecordPackHeader(miss, &records[3]));
    for (uint32_t i = 0; i < kNumHitRecords; ++i) OPTIX_CHECK(optixSbtRecordPackHeader(hitgroup, &records[4 + i]));
    const size_t bytes = records.size() * sizeof(EmptyRecord);
    CUDA_CHECK(cudaMalloc((void**)&sbt_buffer, bytes));
    CUDA_CHECK(cudaMemcpy((void*)sbt_buffer, records.data(), bytes, cudaMemcpyHostToDevice));

    sbt = {};
    sbt.raygenRecord = sbt_buffer;
    sbt.missRecordBase = sbt_buffer + 3 * sizeof(EmptyRecord);
    sbt.missRecordStrideInBytes = sizeof(EmptyRecord);
    sbt.missRecordCount = 1;
    sbt.hitgroupRecordBase = sbt_buffer + 4 * sizeof(EmptyRecord);
    sbt.hitgroupRecordStrideInBytes = sizeof(EmptyRecord);
    sbt.hitgroupRecordCount = kNumHitRecords;
    sbt_guides = sbt;
    sbt_guides.raygenRecord = sbt_buffer + sizeof(EmptyRecord);
    sbt_display = sbt;
    sbt_display.raygenRecord = sbt_buffer + 2 * sizeof(EmptyRecord);
}

OptixState::~OptixState() {
    if (sbt_buffer) cudaFree((void*)sbt_buffer);
    if (pipeline) optixPipelineDestroy(pipeline);
    if (hitgroup) optixProgramGroupDestroy(hitgroup);
    if (miss) optixProgramGroupDestroy(miss);
    if (raygen_display) optixProgramGroupDestroy(raygen_display);
    if (raygen_guides) optixProgramGroupDestroy(raygen_guides);
    if (raygen) optixProgramGroupDestroy(raygen);
    if (module) optixModuleDestroy(module);
    if (context) optixDeviceContextDestroy(context);
}
