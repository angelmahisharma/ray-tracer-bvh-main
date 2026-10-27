// Sanity tests for the shared library. Run: ctest   (or ./build/test_basic <scenes_dir>)
#include <cstdio>
#include <iostream>
#include <string>
#include <type_traits>
#include "rt/image.h"
#include "rt/rng.h"
#include "rt/scene.h"

using namespace rt;

static int failures = 0;
#define CHECK(cond) do { if (!(cond)) { std::cerr << "FAIL line " << __LINE__ << ": " #cond "\n"; ++failures; } } while (0)
static bool near(float a, float b, float eps = 1e-4f) { return fabsf(a - b) < eps; }

int main(int argc, char** argv) {
    std::string dir = argc > 1 ? argv[1] : "scenes";
    if (!dir.empty() && dir.back() != '/') dir += '/';

    // --- GPU-copyable layouts ---
    CHECK(std::is_trivially_copyable<Triangle>::value);
    CHECK(std::is_trivially_copyable<Material>::value);
    CHECK(std::is_trivially_copyable<Camera>::value);
    CHECK(std::is_standard_layout<Triangle>::value);
    CHECK(sizeof(Triangle) == 80 && sizeof(Material) == 48 && sizeof(Light) == 16);

    // --- math ---
    Vec3 x(1, 0, 0), y(0, 1, 0);
    CHECK(near(cross(x, y).z, 1.f));
    CHECK(near(length(Vec3(3, 4, 0)), 5.f));
    CHECK(near(length(normalize(Vec3(2, 3, 4))), 1.f));

    // --- rng: deterministic and in range ---
    Rng a, b;
    a.seed(42, 7); b.seed(42, 7);
    bool same = true, in_range = true;
    double sum = 0;
    for (int i = 0; i < 100000; ++i) {
        float fa = a.next_float(), fb = b.next_float();
        same = same && (fa == fb);
        in_range = in_range && fa >= 0.f && fa < 1.f;
        sum += fa;
    }
    CHECK(same); CHECK(in_range);
    CHECK(fabs(sum / 100000.0 - 0.5) < 0.01);

    // --- camera: the centre ray points at the target ---
    Camera cam = make_camera(Vec3(0, 0, 5), Vec3(0, 0, 0), Vec3(0, 1, 0), 40.f, 16.f / 9.f);
    Ray r = cam.generate_ray(0.5f, 0.5f);
    CHECK(near(r.dir.z, -1.f) && near(r.dir.x, 0.f) && near(r.dir.y, 0.f));
    Ray top = pixel_ray(cam, 0, 0, 100, 100, 0.5f, 0.5f);   // top-left pixel looks up and left
    CHECK(top.dir.y > 0.f && top.dir.x < 0.f);

    // --- scenes ---
    Scene s; std::string err;
    CHECK(load_scene(dir + "one_triangle.scene", s, &err));
    CHECK(s.triangles.size() == 1);
    CHECK(near(length(s.triangles[0].n0), 1.f));

    CHECK(load_scene(dir + "cube.scene", s, &err));
    CHECK(s.triangles.size() == 12);
    for (const Triangle& t : s.triangles)                       // flat normals point outward
        CHECK(dot(t.geometric_normal(), t.centroid()) > 0.f);
    CHECK(near(s.bounds.lo.x, -0.5f) && near(s.bounds.hi.y, 0.5f));
    CHECK(s.flat_positions().size() == 12 * 9);

    CHECK(load_scene(dir + "spheres.scene", s, &err));
    CHECK(s.triangles.size() == 2164);
    CHECK(s.lights.size() == 2 && near(s.lights.back().cdf, 1.f));
    CHECK(s.materials.size() >= 5);

    CHECK(!load_scene(dir + "does_not_exist.scene", s, &err));
    CHECK(!err.empty());

    // --- image round trip ---
    Image img(7, 5);
    for (int yy = 0; yy < 5; ++yy) for (int xx = 0; xx < 7; ++xx) img.set(xx, yy, Color(xx * 0.1f, yy * 0.2f, 2.5f));
    CHECK(save_pfm(img, "_t.pfm"));
    Image back;
    CHECK(load_pfm("_t.pfm", back));
    CHECK(back.width == 7 && back.height == 5 && back.rgb == img.rgb);
    CHECK(save_png(img, "_t.png"));
    std::remove("_t.pfm"); std::remove("_t.png");

    if (failures) { std::cerr << failures << " check(s) failed\n"; return 1; }
    std::cout << "all tests passed\n";
    return 0;
}
