#include "rt/scene.h"
#include <iomanip>
#include <ostream>
#include <sstream>

namespace rt {

void print_scene_stats(const Scene& s, std::ostream& os) {
    size_t per_type[4] = {0, 0, 0, 0};
    for (const Triangle& t : s.triangles) per_type[s.materials[t.material_id].type]++;
    Vec3 e = s.bounds.extent();
    double mb = s.triangle_bytes() / (1024.0 * 1024.0);

    os << "Scene file     : " << s.source_path << "\n";
    os << "Image          : " << s.settings.width << " x " << s.settings.height
       << ", " << s.settings.spp << " spp, max " << s.settings.max_bounces << " bounces\n";
    os << "Triangles      : " << s.triangles.size();
    if (s.degenerate_skipped) os << "  (" << s.degenerate_skipped << " degenerate skipped)";
    os << "\n";
    os << "Materials      : " << s.materials.size() << "\n";
    os << "  triangles by material type: diffuse " << per_type[MAT_DIFFUSE]
       << ", mirror " << per_type[MAT_MIRROR] << ", glass " << per_type[MAT_GLASS]
       << ", emissive " << per_type[MAT_EMISSIVE] << "\n";
    float emissive_area = 0.f;
    for (const Light& l : s.lights) emissive_area += l.area;
    os << "Area lights    : " << s.lights.size() << " triangles, total area "
       << emissive_area << ", total power " << s.total_light_power << "\n";
    os << "Environment    : " << (s.env.mode == ENV_CONSTANT ? "constant" : "gradient") << "\n";
    os << std::fixed << std::setprecision(3);
    os << "Bounds min     : " << s.bounds.lo.x << " " << s.bounds.lo.y << " " << s.bounds.lo.z << "\n";
    os << "Bounds max     : " << s.bounds.hi.x << " " << s.bounds.hi.y << " " << s.bounds.hi.z << "\n";
    os << "Scene size     : " << e.x << " x " << e.y << " x " << e.z
       << "  (diagonal " << length(e) << ")\n";
    os << "Triangle memory: " << s.triangle_bytes() << " bytes (" << std::setprecision(2) << mb << " MiB)\n";
}

std::string scene_stats_csv_header() {
    return "scene,triangles,materials,lights,bounds_diagonal,triangle_bytes";
}

std::string scene_stats_csv_row(const Scene& s) {
    std::ostringstream o;
    o << s.source_path << "," << s.triangles.size() << "," << s.materials.size() << ","
      << s.lights.size() << "," << length(s.bounds.extent()) << "," << s.triangle_bytes();
    return o.str();
}

}  // namespace rt
