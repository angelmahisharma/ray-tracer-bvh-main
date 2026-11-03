#pragma once
#include <cstdint>
#include <vector>
#include <optix.h>
#include "rt/scene.h"

// Triangle geometry acceleration structure (the hardware BVH).
struct Gas {
    OptixTraversableHandle handle = 0;
    CUdeviceptr buffer = 0;
    size_t bytes = 0;          // after compaction
    size_t uncompacted_bytes = 0;
    double build_ms = 0.0;     // build + compaction, excluding the vertex upload

    Gas() = default;
    Gas(const Gas&) = delete;
    Gas& operator=(const Gas&) = delete;
    ~Gas();
};

// sbt_offsets: one hit-group record index per triangle (see kNumHitRecords).
void build_gas(OptixDeviceContext ctx, const rt::Scene& scene,
               const std::vector<uint8_t>& sbt_offsets, Gas& out);
