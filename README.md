# BVH Ray Tracing Project: Part 1 (Foundation & Shared Scene System)

Everything the other 7 people build on: build system, scene format + loader,
GPU-copyable data layout, math helpers, RNG, image saving, scene statistics,
debug scenes and docs. **No external libraries needed** (just a C++17 compiler
and CMake >= 3.16).

## Quick start

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build                       # should print: 100% tests passed

./build/scene_stats scenes/spheres.scene     # triangle count, bounds, lights, memory
./build/debug_view  scenes/spheres.scene out.png --downscale 2   # normal-colour preview
```
Windows (Visual Studio): `cmake -B build` then `cmake --build build --config Release`
(executables end up in `build/Release/`).

## Folder layout

```
include/rt/      shared headers (header-only math + data types)   <- everyone includes these
src/             loader, image I/O, statistics (compiled into rt::common)
tools/           scene_stats, debug_view, gen_debug_scenes.py
tests/           sanity tests (run with ctest)
scenes/          3 tiny debug scenes (+ Part 7 adds the real test scenes here)
apps/
  cpu_renderer/    Part 2 + 3   (CPU path tracer, then BVH)
  cuda_renderer/   Part 4       (GPU brute force)
  optix_renderer/  Part 5 + 6   (OptiX / RT cores)
  benchmark/       Part 8       (runner, image compare, charts)
docs/            SCENE_FORMAT, HOW_TO_PLUG_IN, DATA_LAYOUT, TEAM_RULES
```

## Read next

| You are...            | Read                                   |
|-----------------------|----------------------------------------|
| Anyone                | `docs/TEAM_RULES.md`, `docs/HOW_TO_PLUG_IN.md` |
| Part 7 (scenes)       | `docs/SCENE_FORMAT.md`                 |
| Part 3, 4, 5, 6       | `docs/DATA_LAYOUT.md`                  |

## What is in the library

| Header          | Contents |
|-----------------|----------|
| `vec3.h`        | `Vec3`, dot, cross, normalize, reflect, min/max, lerp |
| `color.h`       | `Color` (= Vec3, linear RGB), luminance, sRGB transfer |
| `ray.h`         | `Ray` (origin, dir, tmin, tmax) |
| `aabb.h`        | `AABB` (expand, surface area, longest axis) - used by Part 3 |
| `triangle.h`    | `Triangle` (80 B, GPU-copyable), `Light` |
| `material.h`    | `Material` (diffuse / mirror / glass / emissive), `Environment` |
| `camera.h`      | `Camera`, `make_camera`, `pixel_ray` |
| `rng.h`         | `Rng` (PCG32) identical on CPU / CUDA / OptiX |
| `scene.h`       | `Scene`, `load_scene`, stats printer, flat upload buffers |
| `image.h`       | `Image`, `save_png`, `save_pfm`, `load_pfm` |

All headers except `scene.h`/`image.h` are usable from `.cu` files and OptiX
device programs (`RT_HD` expands to `__host__ __device__` under nvcc).

## Git workflow (agreed in week 1)

```bash
git init && git add . && git commit -m "Part 1: foundation"
git branch -M main
git remote add origin <your-repo-url> && git push -u origin main
# everyone else:
git checkout -b part2-cpu-renderer     # one branch per person, named partN-topic
```
Merge into `main` through pull requests. **Never change anything in `include/rt/`
without telling the group** (see TEAM_RULES.md).
