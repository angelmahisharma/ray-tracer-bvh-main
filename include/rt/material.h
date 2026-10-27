#pragma once
#include "color.h"

namespace rt {

enum MaterialType : uint32_t {
    MAT_DIFFUSE  = 0,   // matte:  albedo
    MAT_MIRROR   = 1,   // metal:  albedo (tint), fuzz (0 = perfect mirror)
    MAT_GLASS    = 2,   // glass:  ior, albedo (tint, usually 1,1,1)
    MAT_EMISSIVE = 3    // area light: emission (radiance), triangles using it are lights
};

// 48 bytes, plain data: copy the array straight to the GPU.
struct Material {
    uint32_t type = MAT_DIFFUSE;
    float    ior  = 1.5f;
    float    fuzz = 0.f;
    float    pad0_ = 0.f;
    Color    albedo{0.7f, 0.7f, 0.7f};
    float    pad1_ = 0.f;
    Color    emission{0.f, 0.f, 0.f};
    float    pad2_ = 0.f;
};
static_assert(sizeof(Material) == 48, "Material layout changed - update docs/DATA_LAYOUT.md");

// Background / sky light. Up axis is +Y.
enum EnvMode : uint32_t { ENV_CONSTANT = 0, ENV_GRADIENT = 1 };

struct Environment {
    uint32_t mode = ENV_CONSTANT;
    float pad0_[3] = {0.f, 0.f, 0.f};
    Color a{0.8f, 0.8f, 0.8f};   // constant color, or horizon color
    float pad1_ = 0.f;
    Color b{0.8f, 0.8f, 0.8f};   // zenith color (gradient only)
    float pad2_ = 0.f;

    // Radiance arriving from direction `dir` (normalized).
    RT_HD Color eval(const Vec3& dir) const {
        if (mode == ENV_CONSTANT) return a;
        return lerp(a, b, clampf(dir.y, 0.f, 1.f));
    }
};

}  // namespace rt
