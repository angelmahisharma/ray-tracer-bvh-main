// Flat BVH shared by the CPU builder and the CUDA traversal.
// Everything marked RT_HD compiles as plain C++ and as device code, so the
// traversal can be unit-tested on the CPU (see test_bvh.cpp).
#pragma once
#include <vector>
#include "rt/triangle.h"
#include "rt/ray.h"

namespace rt {

// 32 bytes, depth-first order. Interior node: count == 0, left child is the
// next node (index + 1), `offset` is the right child's index. Leaf: count > 0,
// `offset` is the first triangle (triangles are stored in BVH order).
struct BVHNode {
    Vec3 lo;
    uint32_t offset = 0;
    Vec3 hi;
    uint32_t count = 0;
    RT_HD bool is_leaf() const { return count > 0; }
};
static_assert(sizeof(BVHNode) == 32, "BVHNode must be 32 bytes");

constexpr int kBvhStackSize = 64;

struct BVH {
    std::vector<BVHNode>  nodes;
    std::vector<Triangle> tris;        // reordered copy, leaves index into this
    std::vector<uint32_t> orig_index;  // tris[i] == original_triangles[orig_index[i]]
    int max_depth = 0;
    int leaf_count = 0;
    float sah_cost = 0.f;              // expected traversal cost, SAH units
};

// Binned-SAH build (host only, bvh_build.cpp).
BVH build_bvh(const std::vector<Triangle>& triangles, int max_leaf_size = 4);

// ---------------------------------------------------------------------------
// Traversal primitives (host + device)
// ---------------------------------------------------------------------------

// Moller-Trumbore, identical maths to apps/cpu_renderer so images match.
RT_HD inline bool intersect_triangle(const Triangle& tri, const Ray& ray, float tmax,
                                     float& t_out, float& u_out, float& v_out) {
    Vec3 e1 = tri.v1 - tri.v0, e2 = tri.v2 - tri.v0;
    Vec3 pvec = cross(ray.dir, e2);
    float det = dot(e1, pvec);
    if (fabsf(det) < 1e-8f) return false;
    float inv_det = 1.f / det;
    Vec3 tvec = ray.origin - tri.v0;
    float u = dot(tvec, pvec) * inv_det;
    if (u < 0.f || u > 1.f) return false;
    Vec3 qvec = cross(tvec, e1);
    float v = dot(ray.dir, qvec) * inv_det;
    if (v < 0.f || u + v > 1.f) return false;
    float t = dot(e2, qvec) * inv_det;
    if (t <= ray.tmin || t >= tmax) return false;
    t_out = t; u_out = u; v_out = v;
    return true;
}

// Slab test. Returns true if the ray overlaps the box within (tmin, tmax);
// `tnear` is the entry distance (used to order children).
RT_HD inline bool intersect_aabb(const BVHNode& n, const Vec3& o, const Vec3& inv_d,
                                 float tmin, float tmax, float& tnear) {
    float tx0 = (n.lo.x - o.x) * inv_d.x, tx1 = (n.hi.x - o.x) * inv_d.x;
    float ty0 = (n.lo.y - o.y) * inv_d.y, ty1 = (n.hi.y - o.y) * inv_d.y;
    float tz0 = (n.lo.z - o.z) * inv_d.z, tz1 = (n.hi.z - o.z) * inv_d.z;
    float t0 = fmaxf(fmaxf(fminf(tx0, tx1), fminf(ty0, ty1)), fmaxf(fminf(tz0, tz1), tmin));
    float t1 = fminf(fminf(fmaxf(tx0, tx1), fmaxf(ty0, ty1)), fminf(fmaxf(tz0, tz1), tmax));
    tnear = t0;
    // tiny slack so rays grazing a box face (zero-thickness boxes) are not lost
    return t0 <= t1 * 1.0000004f;
}

struct BvhHit { float t, u, v; uint32_t tri; };

// Nearest hit with tmin < t < ray.tmax. `tris` are in BVH order.
RT_HD inline bool bvh_closest_hit(const BVHNode* nodes, const Triangle* tris, const Ray& ray, BvhHit& hit) {
    Vec3 inv_d(1.f / ray.dir.x, 1.f / ray.dir.y, 1.f / ray.dir.z);
    float closest = ray.tmax, tn;
    if (!intersect_aabb(nodes[0], ray.origin, inv_d, ray.tmin, closest, tn)) return false;
    uint32_t stack[kBvhStackSize];
    int sp = 0;
    uint32_t cur = 0;
    bool found = false;
    for (;;) {
        const BVHNode node = nodes[cur];
        if (node.count > 0) {
            for (uint32_t i = node.offset; i < node.offset + node.count; ++i) {
                float t, u, v;
                if (intersect_triangle(tris[i], ray, closest, t, u, v)) {
                    closest = t; hit.t = t; hit.u = u; hit.v = v; hit.tri = i; found = true;
                }
            }
            if (sp == 0) break;
            cur = stack[--sp];
            continue;
        }
        uint32_t l = cur + 1, r = node.offset;
        float tl, tr;
        bool hl = intersect_aabb(nodes[l], ray.origin, inv_d, ray.tmin, closest, tl);
        bool hr = intersect_aabb(nodes[r], ray.origin, inv_d, ray.tmin, closest, tr);
        if (hl && hr) {
            if (tl <= tr) { stack[sp++] = r; cur = l; } else { stack[sp++] = l; cur = r; }
        } else if (hl) cur = l;
        else if (hr) cur = r;
        else {
            if (sp == 0) break;
            cur = stack[--sp];
        }
    }
    return found;
}

// Shadow-ray query: is anything hit with tmin < t < ray.tmax? Stops at the first hit.
RT_HD inline bool bvh_any_hit(const BVHNode* nodes, const Triangle* tris, const Ray& ray) {
    Vec3 inv_d(1.f / ray.dir.x, 1.f / ray.dir.y, 1.f / ray.dir.z);
    float tn;
    if (!intersect_aabb(nodes[0], ray.origin, inv_d, ray.tmin, ray.tmax, tn)) return false;
    uint32_t stack[kBvhStackSize];
    int sp = 0;
    uint32_t cur = 0;
    for (;;) {
        const BVHNode node = nodes[cur];
        if (node.count > 0) {
            for (uint32_t i = node.offset; i < node.offset + node.count; ++i) {
                float t, u, v;
                if (intersect_triangle(tris[i], ray, ray.tmax, t, u, v)) return true;
            }
            if (sp == 0) break;
            cur = stack[--sp];
            continue;
        }
        uint32_t l = cur + 1, r = node.offset;
        float tl, tr;
        bool hl = intersect_aabb(nodes[l], ray.origin, inv_d, ray.tmin, ray.tmax, tl);
        bool hr = intersect_aabb(nodes[r], ray.origin, inv_d, ray.tmin, ray.tmax, tr);
        if (hl && hr) {
            if (tl <= tr) { stack[sp++] = r; cur = l; } else { stack[sp++] = l; cur = r; }
        } else if (hl) cur = l;
        else if (hr) cur = r;
        else {
            if (sp == 0) break;
            cur = stack[--sp];
        }
    }
    return false;
}

}  // namespace rt
