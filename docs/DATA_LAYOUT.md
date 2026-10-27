# Data layout (what you copy to the GPU)

All structs are plain data (`static_assert`ed sizes, trivially copyable), so
`cudaMemcpy` of a `std::vector<T>` works directly. **Do not reorder fields.**

| Struct | Size | Fields |
|---|---|---|
| `Triangle` | 80 B | `v0 v1 v2` (3x Vec3), `n0 n1 n2` (3x Vec3), `material_id` (u32), pad |
| `Material` | 48 B | `type` (u32), `ior`, `fuzz`, pad, `albedo` (Vec3)+pad, `emission` (Vec3)+pad |
| `Light`    | 16 B | `tri_index` (u32), `area`, `power`, `cdf` |
| `Environment` | 48 B | `mode`, pad[3], `a`+pad, `b`+pad |
| `Camera`   | 48 B | `origin`, `lower_left`, `horizontal`, `vertical` (4x Vec3) |
| `Vec3`     | 12 B | `x y z` (NOT 16-byte aligned: when you want `float4`/`float3` alignment on the GPU, convert) |

```cpp
// CUDA upload example
rt::Triangle* d_tris;
cudaMalloc(&d_tris, scene.triangle_bytes());
cudaMemcpy(d_tris, scene.triangles.data(), scene.triangle_bytes(), cudaMemcpyHostToDevice);
```

## Barycentrics and normals
`P = (1-u-v)*v0 + u*v1 + v*v2`. Shading normal = `tri.shading_normal(u, v)`.
OptiX reports barycentrics as `(u, v)` for `v1` and `v2` too, so they match.

## For OptiX (Part 5)
`Scene::flat_positions()` returns 9 floats per triangle (tightly packed, three
vertices per triangle, no index buffer): use `OPTIX_VERTEX_FORMAT_FLOAT3`,
`vertexStrideInBytes = 12`, `numVertices = 3 * triangles`, and no index buffer.
`Scene::flat_material_ids()` gives one material id per triangle for the SBT
lookup. `optixGetPrimitiveIndex()` is the index into `scene.triangles`.

## For the CPU BVH (Part 3)
`Triangle::bounds()` and `centroid()`, and `AABB` (with `surface_area()` for SAH
and `longest_axis()` for the middle split) are ready. Keep the original
`triangles` array unchanged and store a **reordered index list** in your BVH
(or reorder a copy), so the three renderers can still compare triangle ids.
