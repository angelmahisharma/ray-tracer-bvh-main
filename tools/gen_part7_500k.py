#!/usr/bin/env python3
"""Regenerates the three debug scenes in ../scenes (OBJ + .scene).
Run:  python3 tools/gen_debug_scenes.py     (no dependencies)
Also handy as a starting point for making bigger procedural test scenes:
change SPHERE_STACKS / SPHERE_SLICES to scale triangle counts."""
import math, os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scenes", "part7_500k")
SPHERE_STACKS, SPHERE_SLICES = 300, 278

def write(name, text):
    with open(os.path.join(OUT, name), "w") as f:
        f.write(text)


class Obj:
    def __init__(self):
        self.lines, self.nv, self.nn = [], 0, 0

    def usemtl(self, m):
        self.lines.append(f"usemtl {m}")

    def vert(self, p):
        self.lines.append("v %.6f %.6f %.6f" % p); self.nv += 1; return self.nv

    def norm(self, n):
        self.lines.append("vn %.6f %.6f %.6f" % n); self.nn += 1; return self.nn

    def face(self, *idx):
        self.lines.append("f " + " ".join(str(i) for i in idx))

    def text(self):
        return "\n".join(self.lines) + "\n"


def sphere(o, c, r, mat):
    o.usemtl(mat)
    vs = []
    for i in range(SPHERE_STACKS + 1):
        th = math.pi * i / SPHERE_STACKS
        row = []
        for j in range(SPHERE_SLICES):
            ph = 2 * math.pi * j / SPHERE_SLICES
            n = (math.sin(th) * math.cos(ph), math.cos(th), math.sin(th) * math.sin(ph))
            p = (c[0] + r * n[0], c[1] + r * n[1], c[2] + r * n[2])
            row.append((o.vert(p), o.norm(n)))
        vs.append(row)
    for i in range(SPHERE_STACKS):
        for j in range(SPHERE_SLICES):
            j2 = (j + 1) % SPHERE_SLICES
            a, b, c2, d = vs[i][j], vs[i][j2], vs[i + 1][j2], vs[i + 1][j]
            # counter-clockwise seen from outside; skip degenerate pole triangles
            if i != 0:
                o.face(f"{a[0]}//{a[1]}", f"{d[0]}//{d[1]}", f"{b[0]}//{b[1]}")
            if i != SPHERE_STACKS - 1:
                o.face(f"{b[0]}//{b[1]}", f"{d[0]}//{d[1]}", f"{c2[0]}//{c2[1]}")


# ---------------- 1. one triangle ----------------
write("one_triangle.obj", "# one triangle in the z=0 plane, facing +z\nv -1 -0.8 0\nv 1 -0.8 0\nv 0 1 0\nf 1 2 3\n")
write("one_triangle.scene", """# Smallest possible scene: one triangle, constant sky light.
image 640 360
spp 16
max_bounces 3
camera eye 0 0 3 target 0 0 0 up 0 1 0 fov 45
environment constant 0.8 0.8 0.8
material red diffuse 0.75 0.15 0.15
mesh one_triangle.obj material red
""")

# ---------------- 2. cube ----------------
cube = Obj()
V = [(-.5, -.5, -.5), (.5, -.5, -.5), (.5, .5, -.5), (-.5, .5, -.5),
     (-.5, -.5, .5), (.5, -.5, .5), (.5, .5, .5), (-.5, .5, .5)]
for p in V:
    cube.vert(p)
quads = [(5, 6, 7, 8), (2, 1, 4, 3), (6, 2, 3, 7), (1, 5, 8, 4), (8, 7, 3, 4), (1, 2, 6, 5)]  # +z -z +x -x +y -y
for q in quads:
    cube.face(q[0], q[1], q[2]); cube.face(q[0], q[2], q[3])
write("cube.obj", "# unit cube, 12 triangles, no normals in file (loader makes flat ones)\n" + cube.text())
write("cube.scene", """# Unit cube centred on the origin: 12 triangles, flat normals.
image 640 360
spp 16
max_bounces 3
camera eye 1.8 1.4 2.6 target 0 0 0 up 0 1 0 fov 40
environment gradient 0.9 0.9 0.9  0.4 0.6 1.0
material blue diffuse 0.2 0.35 0.8
mesh cube.obj material blue
""")

# ---------------- 3. spheres + floor + light ----------------
sp = Obj()
sphere(sp, (-2.0, 0.8, 0), 0.8, "matte")
sphere(sp, (0.0, 0.8, 0), 0.8, "metal")
sphere(sp, (2.0, 0.8, 0), 0.8, "glass")
sp.usemtl("floor")
f = [sp.vert(p) for p in [(-6, 0, 6), (6, 0, 6), (6, 0, -6), (-6, 0, -6)]]
sp.face(f[0], f[1], f[2]); sp.face(f[0], f[2], f[3])
sp.usemtl("lamp")
l = [sp.vert(p) for p in [(-1, 3.5, -1), (1, 3.5, -1), (1, 3.5, 1), (-1, 3.5, 1)]]
sp.face(l[0], l[1], l[2]); sp.face(l[0], l[2], l[3])
write("spheres.obj", "# 3 UV spheres (matte / metal / glass) + floor + ceiling lamp\n" + sp.text())
write("spheres.scene", """# Three spheres built from triangles, a floor and a small area light.
# Exercises: diffuse, mirror, glass, emissive, smooth normals from the OBJ.
image 1280 720
spp 64
max_bounces 5
camera eye 0 1.6 6 target 0 0.8 0 up 0 1 0 fov 40
environment gradient 0.6 0.65 0.7  0.15 0.25 0.5
material matte diffuse 0.8 0.3 0.25
material metal mirror 0.9 0.9 0.95 fuzz 0.05
material glass glass 1.5
material floor diffuse 0.6 0.6 0.6
material lamp emissive 12 12 12
mesh spheres.obj
""")
print("scenes written to", os.path.abspath(OUT))
