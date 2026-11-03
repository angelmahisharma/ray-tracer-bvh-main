#include "device_scene.h"

using namespace rt;

static float4 f4(const Vec3& v, float w) { return float4{v.x, v.y, v.z, w}; }

DeviceSceneData pack_scene(const Scene& scene) {
    DeviceSceneData d;

    d.materials.reserve(scene.materials.size());
    for (const Material& m : scene.materials) {
        float param = m.type == MAT_GLASS ? m.ior : m.fuzz;
        d.materials.push_back({f4(m.albedo, (float)m.type), f4(m.emission, param)});
    }

    const size_t n = scene.triangles.size();
    d.tris.resize(n);
    d.tri_sbt_offset.resize(n);
    d.tri_to_light.assign(n, -1);
    for (size_t i = 0; i < n; ++i) {
        const Triangle& t = scene.triangles[i];
        d.tris[i] = {oct_encode(t.n0), oct_encode(t.n1), oct_encode(t.n2), t.material_id};
        d.tri_sbt_offset[i] = (uint8_t)scene.materials[t.material_id].type;
    }

    for (size_t i = 0; i < scene.lights.size(); ++i) {
        const Light& l = scene.lights[i];
        const Triangle& t = scene.triangles[l.tri_index];
        float select_pdf = scene.total_light_power > 0.f ? l.power / scene.total_light_power : 0.f;
        d.lights.push_back({f4(t.v0, l.area), f4(t.v1 - t.v0, select_pdf), f4(t.v2 - t.v0, 0.f),
                            f4(scene.materials[t.material_id].emission, 0.f)});
        d.light_cdf.push_back(l.cdf);
        d.tri_to_light[l.tri_index] = (int32_t)i;
    }
    return d;
}
