// debug_view <file.scene> <out.png> [--downscale N]
//
// Tiny brute-force viewer: one ray per pixel, colour = surface normal.
// It exists only to prove that the loader, the camera and the image writer
// work end-to-end. It is NOT the real renderer (that is Part 2).
#include <cstdlib>
#include <cstring>
#include <iostream>
#include "rt/image.h"
#include "rt/scene.h"

using namespace rt;

static bool hit_triangle(const Triangle& t, const Ray& r, float& tt, float& u, float& v) {
    Vec3 e1 = t.v1 - t.v0, e2 = t.v2 - t.v0;
    Vec3 p = cross(r.dir, e2);
    float det = dot(e1, p);
    if (fabsf(det) < 1e-12f) return false;
    float inv = 1.f / det;
    Vec3 s = r.origin - t.v0;
    u = dot(s, p) * inv;
    if (u < 0.f || u > 1.f) return false;
    Vec3 q = cross(s, e1);
    v = dot(r.dir, q) * inv;
    if (v < 0.f || u + v > 1.f) return false;
    tt = dot(e2, q) * inv;
    return tt > r.tmin && tt < r.tmax;
}

int main(int argc, char** argv) {
    if (argc < 3) { std::cerr << "usage: debug_view <file.scene> <out.png> [--downscale N]\n"; return 1; }
    int down = 1;
    for (int i = 3; i + 1 < argc; ++i)
        if (!strcmp(argv[i], "--downscale")) down = std::max(1, atoi(argv[i + 1]));

    Scene sc;
    std::string err;
    if (!load_scene(argv[1], sc, &err)) { std::cerr << "error: " << err << "\n"; return 1; }
    int W = sc.settings.width / down, H = sc.settings.height / down;
    std::cout << "Rendering " << W << "x" << H << " normal view of " << sc.triangles.size() << " triangles...\n";

    Image img(W, H);
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            Ray r = pixel_ray(sc.camera, x, y, W, H, 0.5f, 0.5f);
            float best = r.tmax, bu = 0, bv = 0;
            const Triangle* hit = nullptr;
            for (const Triangle& t : sc.triangles) {
                float tt, u, v;
                Ray rr = r; rr.tmax = best;
                if (hit_triangle(t, rr, tt, u, v)) { best = tt; bu = u; bv = v; hit = &t; }
            }
            Color c;
            if (!hit) c = sc.env.eval(r.dir);
            else if (sc.materials[hit->material_id].type == MAT_EMISSIVE) c = Color(1.f, 1.f, 0.8f);
            else {
                Vec3 n = hit->shading_normal(bu, bv);
                if (dot(n, r.dir) > 0.f) n = -n;
                c = n * 0.5f + Color(0.5f);
            }
            img.set(x, y, c);
        }
    if (!save_png(img, argv[2], /*apply_srgb=*/false)) { std::cerr << "could not write " << argv[2] << "\n"; return 1; }
    std::cout << "Wrote " << argv[2] << "\n";
    return 0;
}
