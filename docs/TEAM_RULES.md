# Team rules (from plan section 2). Defaults are baked into the loader.

| Rule | Value | Where it is enforced |
|---|---|---|
| Scene format | OBJ + `.scene` text file | `load_scene()`; nobody writes their own loader |
| Camera | pinhole, `make_camera` / `pixel_ray` | `include/rt/camera.h` |
| Materials | diffuse, mirror, glass, emissive; sky = constant or gradient | `include/rt/material.h`; exact math in Part 7's frozen spec |
| Image size | 1280 x 720 | `RenderSettings` default (scene may override) |
| Samples per pixel | 64 | `RenderSettings` default |
| Max bounces | 5 | `RenderSettings` default |
| Output | PNG for viewing + **PFM (raw float) for comparison** | `save_png`, `save_pfm` |
| Random numbers | PCG32, `rng.seed(global_seed, pixel_index)` | `include/rt/rng.h` |
| Git | one repo, one branch per person: `partN-topic` | README |
| Frozen files | `include/rt/*` and `docs/SCENE_FORMAT.md` | change only after telling the group |

Conventions every renderer must follow (so images match):

* Units/axes: right-handed, **+Y is up**; camera looks down its own -Z.
* Triangle winding is counter-clockwise from outside. A ray may hit either side
  (flip the normal if `dot(n, ray.dir) > 0`); glass needs to know front vs back.
* Pixel (0,0) is the **top-left**. Pixel centre = (x+0.5, y+0.5).
* Colours are **linear** RGB inside renderers. Gamma/tone mapping is applied only
  when writing the PNG. Always save the PFM too.
* Ray `tmin` = `kRayEps` (1e-4) to avoid self-intersection.
* Light sampling: pick a light from `Scene::lights` using `Light::cdf`.
* Timing (Part 8's rules): time the render only, never loading; BVH build is
  reported separately.
