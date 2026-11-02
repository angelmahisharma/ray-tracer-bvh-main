# optix_renderer
Parts 5 and 6: OptiX hardware BVH + full path tracer

Same path tracer as `apps/cpu_renderer` (diffuse with next-event estimation and
MIS, mirror, glass, emissive, sky), traced on the RT cores with OptiX 9.

## Build
Needs the CUDA toolkit and the OptiX SDK (headers only). CMake looks in
`$OptiX_INSTALL_DIR`, then `C:/ProgramData/NVIDIA Corporation/OptiX SDK *`; if
either is missing the target is skipped and the rest of the project still builds.

```bash
cmake -B build                                  # -DOptiX_INSTALL_DIR=... if not found
cmake --build build --config Release --target optix_renderer
```

## Run
```
optix_renderer <scene.scene> [output_prefix] [spp] [max_bounces] [options]
  --runs N        timed launches after one warm-up (default 5, median reported)
  --count-rays    extra launch that counts traced rays (for Grays/s)
  --no-ser        disable Shader Execution Reordering
  --maxreg N      cap registers per thread (0 = OptiX default)
  --seed S        global RNG seed (default 1234, same as cpu_renderer)
  --denoise       run the OptiX AI denoiser, also write <prefix>_denoised.png/.pfm
  --guide-spp N   jittered samples for the denoiser's albedo/normal guides (default 4)
  --save-guides   also write <prefix>_albedo.png and <prefix>_normal.png
  --upscale2x     path-trace at half width/height, denoiser upscales to full size
  --verbose       OptiX log level 4
```
`Render time` covers the path-tracing launch only (Part 8 contract); the GAS
(BVH) build is reported separately. With `--denoise` the guide pass, the
denoiser and the whole frame are timed too. `<prefix>.pfm` is always the raw,
un-denoised radiance, so image comparisons stay valid.

## Denoiser
* A separate raygen program (`__raygen__guides`) writes the albedo and
  world-space normal guides of the first non-specular surface (through
  mirror/glass chains). Keeping it out of the path tracer means the path
  tracer's speed is unchanged; the pass costs ~0.9 ms at 720p.
* OptiX AOV denoiser model, all memory allocated once (`denoiser.cpp`), so a
  real-time loop only pays the invoke: ~10 ms at 1280x720 on an RTX 4060 Laptop.
* `--upscale2x` uses the UPSCALE2X model (render a quarter of the pixels). On
  the spheres scene at 720p it loses to native resolution at equal frame rate,
  but it should pay off at higher output resolutions or in heavier scenes.

Real-time budget, spheres 1280x720, 5 bounces (display-space RMSE against a 1024 spp reference):

| spp | frame (render+guides+denoise) | fps | RMSE denoised | RMSE raw |
|---|---|---|---|---|
| 4  | 15.2 ms | 65.7 | 0.0110 | 0.0606 |
| 8  | 19.1 ms | 52.3 | 0.0103 | 0.0628 |
| 16 | 26.8 ms | 37.3 | 0.0098 | 0.0541 |
| 24 | 34.5 ms | 28.9 | 0.0095 | 0.0477 |
| 32 | 42.2 ms | 23.7 | 0.0093 | 0.0431 |

## Interactive viewer
```
optix_viewer <scene.scene> [--spp N] [--max-bounces N] [--no-denoise]
```
| Input | Action |
|---|---|
| left drag | look around |
| right / middle drag | pan |
| scroll | move forward / back |
| W A S D, Q / E | move, down / up (Shift = 5x faster) |
| `[` / `]` | halve / double samples per frame (default 4) |
| N | denoiser on / off |
| C | print the camera as a `.scene` line (handy for fixing scene cameras) |
| R | reset the camera |

While the camera is still, frames are averaged into a running mean (progressive
refinement), so the image keeps converging; any camera change restarts it. The
window title shows samples per frame, accumulated frames and the frame time.
The image goes from CUDA straight into an OpenGL pixel buffer (no copy through
the CPU); if CUDA/OpenGL sharing is unavailable it falls back to a host copy.
It is built from the GLFW and glad sources that ship with the OptiX SDK
(`-DOPTIX_BUILD_VIEWER=OFF` to skip it). Measured on the RTX 4060 Laptop at
4 spp/frame with the denoiser: spheres 1280x720 ~67 fps, Cornell 640x640 ~77 fps.

## Code layout
`renderer.{h,cpp}` owns everything for one scene (pipeline, GAS, buffers,
denoiser) and is shared by `main.cpp` (offline renderer) and `viewer.cpp`.
Device programs are in `programs.cu`: `__raygen__path` (path tracer),
`__raygen__guides` (denoiser guides), `__raygen__display` (viewer sRGB output).

## Design (why it is fast)
* **One GAS, no instances**, `PREFER_FAST_TRACE`, compacted. Geometry is
  `DISABLE_ANYHIT`, so the RT cores never call back into shader code.
* **Everything runs in raygen.** Rays are cast with `optixTraverse` and read
  from the hit object, so there are no closest-hit/miss programs, no payload and
  an empty SBT. Shadow rays use `TERMINATE_ON_FIRST_HIT`.
* **Shader Execution Reordering** after every bounce except the camera hit. The
  hit-group record index is the material type (per-triangle SBT offset in the
  GAS), so `optixReorder()` groups threads by material for free.
* **Minimal live state across traces.** Everything live across
  `optixTraverse`/`optixReorder` is saved to memory, so the loop re-reads the
  camera/sky from constant memory, reads the ray back from the hit object, and
  keeps only the 64-bit RNG state (the stream comes from the pixel index).
* **Path regeneration**: a thread starts its next sample as soon as a path ends.
* **Compact shading data** (`launch_params.h`): 16 B per triangle (octahedral
  normals + material id, one 128-bit load), 32 B materials, binary search over
  the light CDF, triangle-to-light table for MIS.

## Checking it against the CPU renderer
Configure with `-DOPTIX_MATCH_CPU=ON` to consume random numbers exactly like
`cpu_renderer` (same rejection loops and draw order). The two PFMs then match
pixel for pixel except on ~1% of pixels, where hardware (watertight) and
Moller-Trumbore intersection disagree at silhouettes. The default build uses
closed-form sampling with the same distributions: same mean, same noise level.
