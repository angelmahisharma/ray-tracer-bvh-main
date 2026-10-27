#pragma once
#include "aabb.h"

namespace rt {

// One triangle, 80 bytes, plain data (copy the whole array to the GPU).
// Winding: counter-clockwise seen from the outside -> geometric normal =
// normalize(cross(v1 - v0, v2 - v0)).
// Barycentrics: P = (1-u-v)*v0 + u*v1 + v*v2   (Moller-Trumbore's u, v).
struct Triangle {
    Vec3 v0, v1, v2;          // positions
    Vec3 n0, n1, n2;          // per-vertex shading normals (always filled, normalized)
    uint32_t material_id = 0; // index into Scene::materials
    uint32_t pad_ = 0;

    RT_HD Vec3 geometric_normal() const { return normalize(cross(v1 - v0, v2 - v0)); }
    RT_HD float area() const { return 0.5f * length(cross(v1 - v0, v2 - v0)); }
    RT_HD Vec3 centroid() const { return (v0 + v1 + v2) * (1.f / 3.f); }
    RT_HD Vec3 shading_normal(float u, float v) const {
        return normalize(n0 * (1.f - u - v) + n1 * u + n2 * v);
    }
    RT_HD AABB bounds() const {
        AABB b; b.expand(v0); b.expand(v1); b.expand(v2); return b;
    }
};
static_assert(sizeof(Triangle) == 80, "Triangle layout changed - update docs/DATA_LAYOUT.md");

// An area light = one emissive triangle. `cdf` is the normalized cumulative
// power, so to pick a light:
//   r = rng.next_float();  pick first light with r <= cdf   (pdf = power/total)
struct Light {
    uint32_t tri_index = 0;
    float area  = 0.f;
    float power = 0.f;   // area * luminance(emission)
    float cdf   = 0.f;
};
static_assert(sizeof(Light) == 16, "Light layout changed");

}  // namespace rt
