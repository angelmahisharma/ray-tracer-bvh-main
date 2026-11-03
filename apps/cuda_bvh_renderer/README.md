# cuda_bvh_renderer
Part 6 bonus: software BVH in plain CUDA ("GPU BVH without RT cores"), the fourth data point next to
CPU+BVH, CUDA brute force (`apps/cuda_renderer`, Part 4) and OptiX (`apps/optix_renderer`).
The BVH is built on the CPU and traversed by a CUDA kernel.
Same path tracer as `apps/cpu_renderer` (diffuse with next-event estimation + MIS, mirror,
glass, emissive, sky), same RNG seeding (`seed(1234, y*w+x)`) and same CLI contract.

## Build
Needs the CUDA toolkit (`nvcc`). Without it the GPU target is skipped, but the BVH library and
`test_cuda_bvh` still build.
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release     # -DCMAKE_CUDA_ARCHITECTURES=89 to override "native"
cmake --build build -j --target cuda_bvh_renderer
```

## Run
```
cuda_bvh_renderer <scene.scene> [output_prefix] [spp] [max_bounces] [options]
  --runs N     timed renders after one warm-up (default 3, median reported)
  --brute      no BVH, test every triangle (baseline for the speed-up)
  --leaf N     max triangles per BVH leaf (default 4)
  --rows N     image rows per kernel launch (default 45; short launches avoid the display watchdog)
  --seed S     global RNG seed (default 1234)
```
`Render time` is the GPU kernels only. BVH build (CPU) and upload are reported separately.

## Files
| File | Purpose |
|---|---|
| `bvh.h` | 32-byte flat node, ray/triangle and ray/box tests, `bvh_closest_hit`, `bvh_any_hit` (host + device) |
| `bvh_build.cpp` | binned-SAH builder (16 bins, depth-first layout, triangles reordered into leaf order) |
| `scene_view.h` | `SceneView` (POD passed to the kernel) and `prepare_scene` (BVH + light remapping) |
| `path_tracer.h` | the path tracer, plain `RT_HD` code, so it also runs on the CPU |
| `render.cu` / `render.cuh` | one thread per pixel, launched in bands of rows |
| `main.cpp` | CLI, upload, timing, PNG/PFM output |
| `test_bvh.cpp` | CPU test: BVH vs brute force on rays and on whole images (`ctest`) |

## Design notes
* Interior node: left child = next node, `offset` = right child; leaf: `offset` = first triangle.
  Traversal tests both children, goes to the nearer one first and keeps a 64-entry stack.
  Depth is capped (median split below level 32) so the stack cannot overflow.
* Shadow rays use a separate any-hit traversal that stops at the first hit.
* Triangles are stored in BVH order; light triangle ids are remapped, and a per-triangle
  `tri_light` table replaces the CPU renderer's linear light search for MIS.
* No `--use_fast_math`, so results track the CPU renderer closely.

## Matching the CPU renderer
`cpu_renderer` passes several `rng.next_float()` calls as arguments of one expression
(`Vec3 p(2*rng.next_float()-1, ...)`, `pixel_ray(..., rng.next_float(), rng.next_float())`). C++ does not
define the order in which arguments are evaluated: g++ goes right to left, nvcc left to right, so a
straight port draws its random numbers in a different order and the images only agree on average.
`path_tracer.h` spells the order out (z, y, x for the sphere sample; jy before jx), which is what g++ does.
Result on Cornell 16 spp: CPU port == `cpu_renderer` bit for bit; GPU vs `cpu_renderer` RMSE 0.0019,
97% of pixels equal to 1e-3 (the rest differ only by float rounding, FMA, `powf`).
