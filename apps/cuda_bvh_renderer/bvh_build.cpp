#include "bvh.h"
#include <algorithm>
#include <numeric>

namespace rt {
namespace {

constexpr int   kBins = 16;
constexpr float kTraversalCost = 0.5f;   // relative to one triangle test = 1
constexpr int   kForcedMedianDepth = 32; // keeps depth (and the traversal stack) bounded

struct Builder {
    const std::vector<Triangle>& src;
    int max_leaf;
    std::vector<AABB> bounds;
    std::vector<Vec3> centroids;
    std::vector<uint32_t> idx;
    BVH& out;
    float cost_sum = 0.f;
    float root_area = 1.f;

    Builder(const std::vector<Triangle>& s, int ml, BVH& o) : src(s), max_leaf(ml), out(o) {}

    uint32_t make_leaf(uint32_t node, uint32_t begin, uint32_t end, int depth) {
        out.nodes[node].offset = begin;
        out.nodes[node].count = end - begin;
        out.leaf_count++;
        out.max_depth = std::max(out.max_depth, depth);
        return node;
    }

    uint32_t build(uint32_t begin, uint32_t end, int depth) {
        uint32_t node = (uint32_t)out.nodes.size();
        out.nodes.emplace_back();
        AABB box, cbox;
        for (uint32_t i = begin; i < end; ++i) { box.expand(bounds[idx[i]]); cbox.expand(centroids[idx[i]]); }
        out.nodes[node].lo = box.lo;
        out.nodes[node].hi = box.hi;
        const uint32_t n = end - begin;
        if (n == 1) return make_leaf(node, begin, end, depth);

        const int axis = cbox.longest_axis();
        const float cmin = cbox.lo[axis], cext = cbox.hi[axis] - cbox.lo[axis];
        uint32_t mid = begin;
        bool split = false;

        if (cext > 0.f && depth < kForcedMedianDepth) {
            // Binned SAH over all three axes.
            float best_cost = 1e30f;
            int best_axis = -1, best_bin = 0;
            for (int a = 0; a < 3; ++a) {
                float ext = cbox.hi[a] - cbox.lo[a];
                if (ext <= 0.f) continue;
                AABB bb[kBins];
                uint32_t cnt[kBins] = {};
                float scale = kBins / ext;
                for (uint32_t i = begin; i < end; ++i) {
                    int b = std::min(kBins - 1, (int)((centroids[idx[i]][a] - cbox.lo[a]) * scale));
                    cnt[b]++; bb[b].expand(bounds[idx[i]]);
                }
                AABB left[kBins - 1];
                uint32_t lcnt[kBins - 1];
                AABB acc; uint32_t c = 0;
                for (int b = 0; b < kBins - 1; ++b) { acc.expand(bb[b]); c += cnt[b]; left[b] = acc; lcnt[b] = c; }
                acc = AABB(); c = 0;
                for (int b = kBins - 1; b >= 1; --b) {
                    acc.expand(bb[b]); c += cnt[b];
                    uint32_t lc = lcnt[b - 1];
                    if (lc == 0 || c == 0) continue;
                    float cost = lc * left[b - 1].surface_area() + c * acc.surface_area();
                    if (cost < best_cost) { best_cost = cost; best_axis = a; best_bin = b; }
                }
            }
            if (best_axis >= 0) {
                float area = std::max(box.surface_area(), 1e-30f);
                float split_cost = kTraversalCost + best_cost / area;
                bool leaf_ok = (int)n <= max_leaf && (float)n <= split_cost;
                if (leaf_ok) return make_leaf(node, begin, end, depth);
                float ext = cbox.hi[best_axis] - cbox.lo[best_axis];
                float scale = kBins / ext;
                auto it = std::partition(idx.begin() + begin, idx.begin() + end, [&](uint32_t i) {
                    int b = std::min(kBins - 1, (int)((centroids[i][best_axis] - cbox.lo[best_axis]) * scale));
                    return b < best_bin;
                });
                mid = (uint32_t)(it - idx.begin());
                split = mid > begin && mid < end;
            }
        }
        if (!split) {
            // Median split (degenerate centroids or depth limit).
            if (n <= (uint32_t)max_leaf && cext <= 0.f) return make_leaf(node, begin, end, depth);
            int a = cext > 0.f ? axis : 0;
            mid = begin + n / 2;
            if (cext > 0.f)
                std::nth_element(idx.begin() + begin, idx.begin() + mid, idx.begin() + end,
                                 [&](uint32_t x, uint32_t y) { return centroids[x][a] < centroids[y][a]; });
        }
        build(begin, mid, depth + 1);
        uint32_t right = build(mid, end, depth + 1);
        out.nodes[node].offset = right;
        out.nodes[node].count = 0;
        return node;
    }
};

// SAH cost of the finished tree (traversal + leaf triangle tests).
float sah_cost(const std::vector<BVHNode>& nodes) {
    if (nodes.empty()) return 0.f;
    AABB root; root.lo = nodes[0].lo; root.hi = nodes[0].hi;
    float root_area = std::max(root.surface_area(), 1e-30f), cost = 0.f;
    for (const BVHNode& nd : nodes) {
        AABB b; b.lo = nd.lo; b.hi = nd.hi;
        float rel = b.surface_area() / root_area;
        cost += nd.count ? rel * nd.count : rel * kTraversalCost;
    }
    return cost;
}

}  // namespace

BVH build_bvh(const std::vector<Triangle>& triangles, int max_leaf_size) {
    BVH out;
    const uint32_t n = (uint32_t)triangles.size();
    if (n == 0) {
        out.nodes.emplace_back();   // empty root: lo > hi never intersects
        out.nodes[0].lo = Vec3(kBigFloat); out.nodes[0].hi = Vec3(-kBigFloat);
        return out;
    }
    Builder b(triangles, std::max(1, max_leaf_size), out);
    b.bounds.resize(n); b.centroids.resize(n); b.idx.resize(n);
    std::iota(b.idx.begin(), b.idx.end(), 0u);
    for (uint32_t i = 0; i < n; ++i) { b.bounds[i] = triangles[i].bounds(); b.centroids[i] = triangles[i].centroid(); }
    out.nodes.reserve(2 * (size_t)n);
    b.build(0, n, 0);
    out.nodes.shrink_to_fit();
    out.tris.resize(n);
    out.orig_index = b.idx;
    for (uint32_t i = 0; i < n; ++i) out.tris[i] = triangles[b.idx[i]];
    out.sah_cost = sah_cost(out.nodes);
    return out;
}

}  // namespace rt
