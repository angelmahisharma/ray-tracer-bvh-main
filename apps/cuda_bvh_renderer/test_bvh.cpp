// CPU-side check of the BVH + path tracer used by the CUDA kernel (no GPU needed).
//   test_bvh <scenes-dir>
#include <cstdio>
#include <string>
#include "path_tracer.h"

using namespace rt;

static int failures = 0;
#define CHECK(cond, ...) do { if (!(cond)) { std::printf("FAIL: " __VA_ARGS__); std::printf("\n"); ++failures; } } while (0)

static bool test_scene(const std::string& path, int rays) {
    Scene scene; std::string err;
    if (!load_scene(path, scene, &err)) { std::printf("load %s: %s\n", path.c_str(), err.c_str()); return false; }
    PreparedScene ps; prepare_scene(scene, ps, 4);
    SceneView bv = ps.host_view(true), pv = ps.host_view(false);

    // structural: every triangle appears exactly once, depth fits the traversal stack
    std::vector<int> seen(scene.triangles.size(), 0);
    for (const BVHNode& n : ps.bvh.nodes)
        if (n.count) for (uint32_t i = n.offset; i < n.offset + n.count; ++i) seen[i]++;
    for (int s : seen) CHECK(s == 1, "%s: triangle referenced %d times", path.c_str(), s);
    CHECK(ps.bvh.max_depth < kBvhStackSize - 4, "%s: depth %d too deep", path.c_str(), ps.bvh.max_depth);

    Rng rng; rng.seed(7, 0);
    Vec3 c = scene.bounds.center(), e = scene.bounds.extent();
    float r = length(e);
    int mismatches = 0, hits = 0, any_bad = 0;
    for (int i = 0; i < rays; ++i) {
        Vec3 o = c + Vec3(rng.next_float() - 0.5f, rng.next_float() - 0.5f, rng.next_float() - 0.5f) * (2.f * r);
        Vec3 target = c + Vec3(rng.next_float() - 0.5f, rng.next_float() - 0.5f, rng.next_float() - 0.5f) * r;
        Ray ray(o, normalize(target - o), kRayEps, (i % 3 == 0) ? r * 0.7f : kBigFloat);
        BvhHit a, b;
        bool fa = bvh_closest_hit(bv.nodes, bv.tris, ray, a), fb = brute_closest(pv, ray, b);
        hits += fa;
        if (fa != fb || (fa && fabsf(a.t - b.t) > 1e-4f * fmaxf(1.f, b.t))) ++mismatches;
        if (bvh_any_hit(bv.nodes, bv.tris, ray) != fb) ++any_bad;
    }
    CHECK(mismatches == 0, "%s: %d/%d closest-hit mismatches", path.c_str(), mismatches, rays);
    CHECK(any_bad == 0, "%s: %d any-hit mismatches", path.c_str(), any_bad);

    // path-traced image: BVH vs brute force must agree (same RNG streams)
    SceneView b2 = bv, p2 = pv;
    b2.width = p2.width = 48; b2.height = p2.height = 27; b2.spp = p2.spp = 8;
    Camera cam = scene.camera;  // aspect differs slightly; irrelevant for the comparison
    b2.camera = p2.camera = cam;
    double diff = 0, mean = 0; int n = 0;
    for (int y = 0; y < 27; ++y) for (int x = 0; x < 48; ++x) {
        Color a = render_pixel(b2, x, y), b = render_pixel(p2, x, y);
        diff += fabs(a.x - b.x) + fabs(a.y - b.y) + fabs(a.z - b.z);
        mean += b.x + b.y + b.z; ++n;
    }
    std::printf("%-40s tris=%-7zu nodes=%-7zu depth=%-2d sah=%.1f  hits %d/%d  img diff/mean = %.5f\n",
                path.c_str(), scene.triangles.size(), ps.bvh.nodes.size(), ps.bvh.max_depth, ps.bvh.sah_cost,
                hits, rays, mean > 0 ? diff / mean : 0.0);
    CHECK(mean > 0, "%s: image is black", path.c_str());
    CHECK(diff <= 0.002 * mean, "%s: BVH image differs from brute-force image", path.c_str());
    return true;
}

int main(int argc, char** argv) {
    std::string dir = argc > 1 ? argv[1] : "scenes";
    const char* scenes[] = {"one_triangle.scene", "cube.scene", "spheres.scene", "part7_1k/spheres.scene",
                            "part7_cornell/cornell.scene", "part7_model/model.scene"};
    for (const char* s : scenes) test_scene(dir + "/" + s, 20000);
    if (argc > 2) test_scene(dir + "/part7_50k/spheres.scene", 5000);
    if (failures) { std::printf("%d failure(s)\n", failures); return 1; }
    std::printf("all BVH tests passed\n");
    return 0;
}
