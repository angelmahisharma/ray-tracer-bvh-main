// Data shared by the host code and the OptiX device programs (programs.cu).
// Everything here is plain data built from rt::Scene by device_scene.cpp and
// laid out for the GPU: 16-byte aligned records, one 128-bit load per access.
#pragma once
#include <cstdint>
#include <optix_types.h>
#include <vector_types.h>
#include "rt/vec3.h"

// Per-triangle shading record: 16 B -> exactly one 128-bit load per hit.
// Positions are NOT stored here; they only live inside the GAS.
struct TriShade {
    uint32_t n0, n1, n2;   // shading normals, octahedral-encoded (2 x snorm16)
    uint32_t material;     // index into LaunchParams::materials
};
static_assert(sizeof(TriShade) == 16, "TriShade must stay one 128-bit load");

// 32 B (rt::Material is 48 B).
struct DevMaterial {
    float4 albedo_type;      // xyz = albedo, w = (float)rt::MaterialType
    float4 emission_param;   // xyz = emission, w = fuzz (mirror) or ior (glass)
};
static_assert(sizeof(DevMaterial) == 32, "DevMaterial layout");

// One emissive triangle, everything next-event estimation needs in 64 B.
struct DevLight {
    float4 v0_area;    // xyz = v0,      w = area
    float4 e1_pdf;     // xyz = v1 - v0, w = select pdf (power / total power)
    float4 e2;         // xyz = v2 - v0
    float4 emission;   // xyz = radiance
};
static_assert(sizeof(DevLight) == 64, "DevLight layout");

// SBT hit-group record index = material type, so optixReorder() groups
// threads by material type without any extra coherence-hint work.
constexpr uint32_t kNumHitRecords = 4;

// Lives in constant memory on the device (every thread reads the same values).
// Plain float3s instead of rt::Camera/rt::Environment: __constant__ variables
// cannot have types with default member initializers.
struct LaunchParams {
    OptixTraversableHandle handle;
    float4*             accum;          // width*height, final pixel value
    const TriShade*     tris;
    const DevMaterial*  materials;
    const DevLight*     lights;
    const float*        light_cdf;      // separate so the binary search only touches floats
    const int32_t*      tri_to_light;   // -1 when the triangle is not a light
    unsigned long long* ray_counter;    // null unless --count-rays
    float4*             albedo;         // denoiser guides, written by __raygen__guides
    float4*             normal;         //   (world-space normals, as the AOV denoiser expects)
    const float4*       display_src;    // __raygen__display: linear image to show
    uchar4*             display;        //   8-bit sRGB output, bottom row first (OpenGL)

    float3 cam_origin, cam_lower_left, cam_horizontal, cam_vertical;
    float3 env_a, env_b;
    uint32_t env_mode;

    uint32_t width, height, spp, max_bounces;
    uint32_t num_lights;
    uint32_t seed;
    uint32_t reorder;                   // 1 = Shader Execution Reordering before shading
    uint32_t guide_spp;                 // jittered samples per pixel for the guide pass
    uint32_t frame_index;               // > 0: average this launch into accum (progressive viewer)
};

// ---- octahedral normal encoding (unit vector <-> 2 x snorm16 in one uint32) ----

RT_HD inline uint32_t oct_snorm16(float v) {
    return (uint32_t)(uint16_t)(int16_t)roundf(rt::clampf(v, -1.f, 1.f) * 32767.f);
}

RT_HD inline uint32_t oct_encode(const rt::Vec3& n) {
    float l1 = fabsf(n.x) + fabsf(n.y) + fabsf(n.z);
    if (l1 <= 0.f) return oct_snorm16(0.f) | (oct_snorm16(0.f) << 16);   // decodes to +Z
    float px = n.x / l1, py = n.y / l1;
    if (n.z < 0.f) {
        float ox = (1.f - fabsf(py)) * (px >= 0.f ? 1.f : -1.f);
        float oy = (1.f - fabsf(px)) * (py >= 0.f ? 1.f : -1.f);
        px = ox; py = oy;
    }
    return oct_snorm16(px) | (oct_snorm16(py) << 16);
}

RT_HD inline rt::Vec3 oct_decode(uint32_t e) {
    float px = (float)(int16_t)(e & 0xffffu) * (1.f / 32767.f);
    float py = (float)(int16_t)(e >> 16) * (1.f / 32767.f);
    rt::Vec3 v(px, py, 1.f - fabsf(px) - fabsf(py));
    float t = fmaxf(-v.z, 0.f);
    v.x += v.x >= 0.f ? -t : t;
    v.y += v.y >= 0.f ? -t : t;
    return rt::normalize(v);
}
