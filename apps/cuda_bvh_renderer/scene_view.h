// What the GPU kernel (and the CPU test) sees: raw pointers + small POD values.
// `PreparedScene` owns the host copy; main.cpp swaps the pointers for device ones.
#pragma once
#include <vector>
#include "bvh.h"
#include "rt/scene.h"

namespace rt {

struct SceneView {
    const BVHNode*   nodes = nullptr;
    const Triangle*  tris = nullptr;       // BVH order (or original order with use_bvh == 0)
    const Material*  materials = nullptr;
    const Light*     lights = nullptr;     // Light::tri_index refers to `tris`
    const int32_t*   tri_light = nullptr;  // per triangle: index into lights, or -1
    uint32_t num_tris = 0, num_materials = 0, num_lights = 0;
    float total_light_power = 0.f;
    Environment env;
    Camera camera;
    int width = 0, height = 0, spp = 0, max_bounces = 0;
    uint64_t seed = 1234;
    int use_bvh = 1;
};

struct PreparedScene {
    BVH bvh;                          // tris in BVH order
    std::vector<Triangle> plain_tris; // original order, for --brute
    std::vector<Material> materials;
    std::vector<Light> lights;        // BVH-order triangle indices
    std::vector<Light> plain_lights;  // original-order triangle indices
    std::vector<int32_t> tri_light, plain_tri_light;
    SceneView view;                   // host pointers into the vectors above

    SceneView host_view(bool use_bvh) const {
        SceneView v = view;
        v.use_bvh = use_bvh;
        v.tris = use_bvh ? bvh.tris.data() : plain_tris.data();
        v.lights = use_bvh ? lights.data() : plain_lights.data();
        v.tri_light = use_bvh ? tri_light.data() : plain_tri_light.data();
        return v;
    }
};

inline void prepare_scene(const Scene& scene, PreparedScene& ps, int max_leaf_size = 4) {
    ps.bvh = build_bvh(scene.triangles, max_leaf_size);
    ps.plain_tris = scene.triangles;
    ps.materials = scene.materials;
    ps.plain_lights = scene.lights;
    ps.lights = scene.lights;
    ps.plain_tri_light.assign(scene.triangles.size(), -1);
    ps.tri_light.assign(scene.triangles.size(), -1);
    for (size_t i = 0; i < scene.lights.size(); ++i)
        if (scene.lights[i].tri_index < scene.triangles.size()) ps.plain_tri_light[scene.lights[i].tri_index] = (int32_t)i;
    // original triangle id -> BVH slot
    std::vector<uint32_t> slot(scene.triangles.size());
    for (size_t i = 0; i < ps.bvh.orig_index.size(); ++i) slot[ps.bvh.orig_index[i]] = (uint32_t)i;
    for (size_t i = 0; i < ps.lights.size(); ++i) {
        if (ps.lights[i].tri_index >= slot.size()) continue;
        ps.lights[i].tri_index = slot[ps.lights[i].tri_index];
        ps.tri_light[ps.lights[i].tri_index] = (int32_t)i;
    }
    SceneView& v = ps.view;
    v.nodes = ps.bvh.nodes.data();
    v.materials = ps.materials.data();
    v.num_tris = (uint32_t)scene.triangles.size();
    v.num_materials = (uint32_t)ps.materials.size();
    v.num_lights = (uint32_t)ps.lights.size();
    v.total_light_power = scene.total_light_power;
    v.env = scene.env;
    v.camera = scene.camera;
    v.width = scene.settings.width; v.height = scene.settings.height;
    v.spp = scene.settings.spp; v.max_bounces = scene.settings.max_bounces;
}

}  // namespace rt
