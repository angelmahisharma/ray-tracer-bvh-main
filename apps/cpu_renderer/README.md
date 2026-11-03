# cpu_renderer
Parts 2 and 3: CPU path tracer, then BVH build/traversal

Put your sources here and add a `CMakeLists.txt` (template in docs/HOW_TO_PLUG_IN.md).
The top-level build adds this folder automatically once the file exists.

## CPU denoising (OIDN)

The CPU renderer can optionally denoise its linear RGB result with Intel Open
Image Denoise (OIDN) 2.5.1. The official Windows binary package is installed in
`third_party/oidn` locally (it is intentionally excluded from Git). Configure
with a 64-bit C++ toolchain so CMake can find that package, then build
`cpu_renderer`. On Windows, OIDN runtime DLLs are copied beside the executable.

Run with the existing positional arguments followed by `--denoise`:

```text
cpu_renderer scene.scene output 4 4 sah --denoise
```

OpenMP parallelizes the image rows when available. Limit or select the worker
count with `--threads N`, for example `--denoise --threads 8`.

This keeps `output.png` and `output.pfm` as the original render and writes
`output_denoised.png` and `output_denoised.pfm`. OIDN receives anti-aliased,
per-pixel first-hit albedo and shading-normal guides accumulated over the same
camera samples as the beauty image. Diffuse surfaces use material albedo,
mirrors use their tint, glass uses neutral albedo, and emissive surfaces use
zero albedo.
