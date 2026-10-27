# How to plug in (1-page guide)

## 1. Link the library

Put your code in your `apps/<yourfolder>/` and add a `CMakeLists.txt`; the top
level picks it up automatically:

```cmake
# apps/cpu_renderer/CMakeLists.txt
add_executable(cpu_renderer main.cpp)
target_link_libraries(cpu_renderer PRIVATE rt::common)
find_package(OpenMP)                       # optional, for multithreading
if(OpenMP_CXX_FOUND)
  target_link_libraries(cpu_renderer PRIVATE OpenMP::OpenMP_CXX)
endif()
```
CUDA targets: `project(... LANGUAGES CXX CUDA)` is already compatible; enable it
in your own `apps/cuda_renderer/CMakeLists.txt` with `enable_language(CUDA)` and
use `.cu` files. Include the headers the same way (`#include "rt/vec3.h"`).

## 2. Load a scene

```cpp
#include "rt/scene.h"
#include "rt/image.h"
#include "rt/rng.h"
using namespace rt;

Scene scene; std::string err;
if (!load_scene(argv[1], scene, &err)) { fprintf(stderr, "%s\n", err.c_str()); return 1; }

const int W = scene.settings.width, H = scene.settings.height;
Image img(W, H);
for (int y = 0; y < H; ++y)
  for (int x = 0; x < W; ++x) {
    Rng rng; rng.seed(/*global_seed=*/1234, /*stream=*/(uint64_t)y * W + x);
    Color sum(0.f);
    for (int s = 0; s < scene.settings.spp; ++s) {
        Ray ray = pixel_ray(scene.camera, x, y, W, H, rng.next_float(), rng.next_float());
        sum += trace(scene, ray, rng);         // <- your path tracer
    }
    img.set(x, y, sum / (float)scene.settings.spp);
  }
save_pfm(img, "out.pfm");    // raw float, for comparisons (always save this)
save_png(img, "out.png");    // for looking at
```

What `scene` gives you:

| Member | Meaning |
|---|---|
| `triangles` | the geometry (+ per-vertex normals + `material_id`) |
| `materials` | indexed by `triangle.material_id` |
| `lights` | emissive triangles with area, power, `cdf` for light picking |
| `env` | `env.eval(dir)` = sky radiance for rays that miss |
| `camera`, `settings` | camera and w/h/spp/bounces (command line may override) |
| `bounds` | scene `AABB` |

## 3. Per-part notes

* **Part 2:** keep a single function `bool closest_hit(const Scene&, const Ray&, Hit&)`
  so Part 3 can swap it for the BVH. Use `kRayEps` as tmin.
* **Part 3:** the library gives you `AABB` and `Triangle::bounds()`; see DATA_LAYOUT.md.
* **Part 4:** upload `scene.triangles`, `scene.materials`, `scene.lights`, `scene.env`,
  `scene.camera` with `cudaMemcpy`; use `Rng` on the device (same header).
* **Part 5/6:** see the OptiX section in DATA_LAYOUT.md.
* **Part 7:** put new scenes in `scenes/` as `.scene` + `.obj`; check them with
  `./build/scene_stats` and `./build/debug_view` before sharing.
* **Part 8:** `./build/scene_stats scene.scene --csv` gives table rows;
  `load_pfm()` reads any renderer's float output for RMSE/SSIM.

## 4. Handy tools

```bash
./build/scene_stats scenes/spheres.scene          # report
./build/scene_stats scenes/spheres.scene --csv    # one CSV row
./build/debug_view  scenes/spheres.scene out.png --downscale 2
python3 tools/gen_debug_scenes.py                 # regenerate debug scenes
```

## 5. Found a bug / need a change in `include/rt/`?
Message Part 1's owner. Do not edit shared headers on your own branch.
