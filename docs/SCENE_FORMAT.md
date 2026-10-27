# Scene format

A scene is one small text file (`*.scene`) plus one or more Wavefront `.obj`
files. Lines starting with `#` (or text after `#`) are comments. Blank lines are
ignored. Paths are relative to the `.scene` file.

## .scene commands

```
image W H                     # default 1280 720
spp N                         # default 64
max_bounces N                 # default 5

camera eye X Y Z target X Y Z [up X Y Z] [fov DEG]     # vertical fov, default up 0 1 0, fov 45

environment constant R G B                 # uniform sky
environment gradient R G B  R G B          # horizon colour, then zenith colour (+Y up)

material NAME diffuse  R G B
material NAME mirror   R G B [fuzz F]     # fuzz 0..1, 0 = perfect mirror
material NAME glass    IOR [R G B]        # optional tint (default 1 1 1)
material NAME emissive R G B               # radiance; can be > 1. Triangles with it are area lights

mesh FILE.obj [material NAME] [scale S] [rotate_y DEG] [translate X Y Z] [smooth]
```

* Define materials **before** the `mesh` line that uses them.
* Material 0 is always `default` (grey diffuse 0.7), used when nothing else is given.
* `camera` and at least one `mesh` are required.
* Transform order for `mesh`: scale -> rotate_y -> translate. Scale must be > 0.
* `smooth`: for faces without normals in the OBJ, compute smooth (area-weighted
  vertex) normals instead of flat ones. Use it for scanned models (bunny, dragon).

## OBJ support

`v`, `vn`, `f` (any of `a`, `a/b`, `a//c`, `a/b/c`; negative indices; polygons
are triangulated as fans) and `usemtl NAME`. `usemtl` picks a material by the
name defined in the `.scene` file (so one OBJ can hold several materials). Unknown
names fall back to the mesh's material with a warning. `vt`, `mtllib`, `o`, `g`,
`s` are ignored. Zero-area triangles are dropped (counted in the stats).

## Examples

See `scenes/*.scene`. Regenerate the debug scenes with `python3 tools/gen_debug_scenes.py`.
For bigger scenes (Part 7) raise `SPHERE_STACKS/SPHERE_SLICES` in that script, or
use downloaded OBJ models:

```
material marble diffuse 0.8 0.8 0.8
mesh bunny.obj material marble scale 10 translate 0 0 0 smooth
```
