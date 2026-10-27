#pragma once
#include <string>
#include <vector>
#include <iosfwd>
#include "camera.h"
#include "material.h"
#include "triangle.h"

namespace rt {

// Team-wide defaults (docs/TEAM_RULES.md). Scene files may override them.
struct RenderSettings {
    int width = 1280;
    int height = 720;
    int spp = 64;           // samples (camera rays) per pixel
    int max_bounces = 5;
};

// Everything a renderer needs, fully loaded and ready to copy to the GPU.
struct Scene {
    std::string source_path;

    RenderSettings settings;
    Camera camera;
    Vec3 cam_eye, cam_target, cam_up{0.f, 1.f, 0.f};   // kept for reference
    float cam_fov = 45.f;
    Environment env;

    std::vector<Material>    materials;       // [0] is always "default"
    std::vector<std::string> material_names;  // same order as materials
    std::vector<Triangle>    triangles;
    std::vector<Light>       lights;          // one per emissive triangle
    float total_light_power = 0.f;

    AABB bounds;                              // of all triangles
    int degenerate_skipped = 0;               // zero-area triangles dropped on load

    // ---- helpers for GPU / OptiX upload ----
    // 9 floats per triangle (v0.xyz v1.xyz v2.xyz) - tightly packed vertex buffer.
    std::vector<float>    flat_positions() const;
    // One material id per triangle (OptiX SBT index / hit-group lookup).
    std::vector<uint32_t> flat_material_ids() const;
    size_t triangle_bytes() const { return triangles.size() * sizeof(Triangle); }
};

// Load a .scene file (which references .obj files). Returns false and fills
// `error` on failure. Format: docs/SCENE_FORMAT.md
bool load_scene(const std::string& scene_path, Scene& out, std::string* error);

// Human readable report (triangle count, bounds, materials, lights, memory).
void print_scene_stats(const Scene& scene, std::ostream& os);
// One CSV line per scene, for the benchmark tables.
std::string scene_stats_csv_header();
std::string scene_stats_csv_row(const Scene& scene);

}  // namespace rt
