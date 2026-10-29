
import math
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "scenes", "part7_cornell")
os.makedirs(OUT, exist_ok=True)


def write_obj(name, triangles):
    path = os.path.join(OUT, name + ".obj")
    with open(path, "w", encoding="utf-8") as f:
        vertex_count = 0
        for tri in triangles:
            for x, y, z in tri:
                f.write(f"v {x} {y} {z}\n")
                vertex_count += 1
        for i in range(1, vertex_count + 1, 3):
            f.write(f"f {i} {i+1} {i+2}\n")
    print("Created:", path)


def quad(a, b, c, d):
    return [(a, b, c), (a, c, d)]


def sphere(cx, cy, cz, radius, stacks=24, slices=32):
    triangles = []

    def point(i, j):
        theta = math.pi * i / stacks
        phi = 2 * math.pi * j / slices
        return (
            cx + radius * math.sin(theta) * math.cos(phi),
            cy + radius * math.cos(theta),
            cz + radius * math.sin(theta) * math.sin(phi),
        )

    for i in range(stacks):
        for j in range(slices):
            p00 = point(i, j)
            p01 = point(i, j + 1)
            p10 = point(i + 1, j)
            p11 = point(i + 1, j + 1)

            if i != 0:
                triangles.append((p00, p10, p01))
            if i != stacks - 1:
                triangles.append((p01, p10, p11))

    return triangles


# Room dimensions: x from -2 to 2, y from 0 to 4,
# and z from -4 (back) to 0 (front).
write_obj("floor", quad(
    (-2, 0, -4), (2, 0, -4), (2, 0, 0), (-2, 0, 0)
))
write_obj("ceiling", quad(
    (-2, 4, -4), (-2, 4, 0), (2, 4, 0), (2, 4, -4)
))
write_obj("back_wall", quad(
    (-2, 0, -4), (-2, 4, -4), (2, 4, -4), (2, 0, -4)
))
write_obj("left_wall", quad(
    (-2, 0, 0), (-2, 4, 0), (-2, 4, -4), (-2, 0, -4)
))
write_obj("right_wall", quad(
    (2, 0, -4), (2, 4, -4), (2, 4, 0), (2, 0, 0)
))

# Small rectangular emissive panel on the ceiling.
write_obj("light", quad(
    (-0.65, 3.98, -2.7), (0.65, 3.98, -2.7),
    (0.65, 3.98, -1.5), (-0.65, 3.98, -1.5)
))

# Two spheres, positioned apart so both can be seen.
write_obj("glass_sphere", sphere(-0.8, 0.8, -2.3, 0.8))
write_obj("mirror_sphere", sphere(0.85, 0.65, -1.6, 0.65))

print("Cornell box OBJ files generated.")
