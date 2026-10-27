#include "rt/scene.h"
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>

namespace rt {
namespace {

std::vector<std::string> split_tokens(const std::string& line) {
    std::string s = line.substr(0, line.find('#'));   // strip comments
    std::istringstream ss(s);
    std::vector<std::string> t;
    std::string w;
    while (ss >> w) t.push_back(w);
    return t;
}

bool parse_float(const std::string& s, float& out) {
    char* end = nullptr;
    out = strtof(s.c_str(), &end);
    return end != s.c_str() && *end == '\0';
}

std::string dir_of(const std::string& path) {
    size_t p = path.find_last_of("/\\");
    return p == std::string::npos ? std::string() : path.substr(0, p + 1);
}

bool is_absolute(const std::string& p) {
    return !p.empty() && (p[0] == '/' || p[0] == '\\' || (p.size() > 1 && p[1] == ':'));
}

class Parser {
public:
    Parser(Scene& s, std::string* e) : sc(s), err(e) {}

    bool parse(const std::string& path) {
        sc = Scene();
        sc.source_path = path;
        base_dir = dir_of(path);
        // Material 0 is always a neutral grey "default".
        sc.materials.push_back(Material());
        sc.material_names.push_back("default");

        std::ifstream in(path);
        if (!in) { cur_file = path; line_no = 0; return bad("cannot open scene file"); }
        cur_file = path;
        std::string line;
        line_no = 0;
        while (std::getline(in, line)) {
            ++line_no;
            auto t = split_tokens(line);
            if (t.empty()) continue;
            if (!handle(t)) return false;
        }
        cur_file = path; line_no = 0;
        if (!have_camera) return bad("scene has no 'camera' line");
        if (sc.triangles.empty()) return bad("scene has no triangles (missing 'mesh' line?)");
        finish();
        return true;
    }

private:
    Scene& sc;
    std::string* err;
    std::string base_dir, cur_file;
    int line_no = 0;
    bool have_camera = false;

    bool bad(const std::string& msg) {
        if (err) *err = cur_file + ":" + std::to_string(line_no) + ": " + msg;
        return false;
    }
    bool nums(const std::vector<std::string>& t, size_t i, int n, float* out) {
        if (i + n > t.size()) return bad("expected " + std::to_string(n) + " number(s) after '" + t[i - 1] + "'");
        for (int k = 0; k < n; ++k)
            if (!parse_float(t[i + k], out[k])) return bad("not a number: '" + t[i + k] + "'");
        return true;
    }
    int find_material(const std::string& name) const {
        for (size_t i = 0; i < sc.material_names.size(); ++i)
            if (sc.material_names[i] == name) return (int)i;
        return -1;
    }

    bool handle(const std::vector<std::string>& t) {
        const std::string& cmd = t[0];
        float v[6];
        if (cmd == "image") {
            if (!nums(t, 1, 2, v)) return false;
            sc.settings.width = (int)v[0]; sc.settings.height = (int)v[1];
            if (sc.settings.width <= 0 || sc.settings.height <= 0) return bad("image size must be positive");
        } else if (cmd == "spp") {
            if (!nums(t, 1, 1, v)) return false;
            sc.settings.spp = (int)v[0];
        } else if (cmd == "max_bounces") {
            if (!nums(t, 1, 1, v)) return false;
            sc.settings.max_bounces = (int)v[0];
        } else if (cmd == "camera") {
            return parse_camera(t);
        } else if (cmd == "environment") {
            return parse_env(t);
        } else if (cmd == "material") {
            return parse_material(t);
        } else if (cmd == "mesh") {
            return parse_mesh(t);
        } else {
            return bad("unknown command '" + cmd + "'");
        }
        return true;
    }

    bool parse_camera(const std::vector<std::string>& t) {
        float v[3];
        bool eye = false, target = false;
        sc.cam_up = Vec3(0.f, 1.f, 0.f);
        for (size_t i = 1; i < t.size();) {
            const std::string& k = t[i++];
            if (k == "eye")         { if (!nums(t, i, 3, v)) return false; sc.cam_eye = Vec3(v[0], v[1], v[2]); i += 3; eye = true; }
            else if (k == "target") { if (!nums(t, i, 3, v)) return false; sc.cam_target = Vec3(v[0], v[1], v[2]); i += 3; target = true; }
            else if (k == "up")     { if (!nums(t, i, 3, v)) return false; sc.cam_up = Vec3(v[0], v[1], v[2]); i += 3; }
            else if (k == "fov")    { if (!nums(t, i, 1, v)) return false; sc.cam_fov = v[0]; i += 1; }
            else return bad("unknown camera option '" + k + "'");
        }
        if (!eye || !target) return bad("camera needs 'eye' and 'target'");
        have_camera = true;
        return true;
    }

    bool parse_env(const std::vector<std::string>& t) {
        float v[6];
        if (t.size() < 2) return bad("environment needs 'constant' or 'gradient'");
        if (t[1] == "constant") {
            if (!nums(t, 2, 3, v)) return false;
            sc.env.mode = ENV_CONSTANT; sc.env.a = Color(v[0], v[1], v[2]); sc.env.b = sc.env.a;
        } else if (t[1] == "gradient") {
            if (!nums(t, 2, 6, v)) return false;
            sc.env.mode = ENV_GRADIENT; sc.env.a = Color(v[0], v[1], v[2]); sc.env.b = Color(v[3], v[4], v[5]);
        } else return bad("environment type must be 'constant' or 'gradient'");
        return true;
    }

    bool parse_material(const std::vector<std::string>& t) {
        float v[3];
        if (t.size() < 3) return bad("material <name> <type> ...");
        const std::string& name = t[1];
        if (find_material(name) >= 0) return bad("material '" + name + "' already defined");
        Material m;
        const std::string& type = t[2];
        if (type == "diffuse") {
            if (!nums(t, 3, 3, v)) return false;
            m.type = MAT_DIFFUSE; m.albedo = Color(v[0], v[1], v[2]);
        } else if (type == "mirror") {
            if (!nums(t, 3, 3, v)) return false;
            m.type = MAT_MIRROR; m.albedo = Color(v[0], v[1], v[2]);
            if (t.size() >= 7) {
                float f;
                if (t[6] != "fuzz" || t.size() < 8 || !parse_float(t[7], f)) return bad("expected 'fuzz <value>'");
                m.fuzz = clampf(f, 0.f, 1.f);
            }
        } else if (type == "glass") {
            float ior;
            if (t.size() < 4 || !parse_float(t[3], ior)) return bad("glass needs an index of refraction");
            m.type = MAT_GLASS; m.ior = ior; m.albedo = Color(1.f, 1.f, 1.f);
            if (t.size() >= 7) { if (!nums(t, 4, 3, v)) return false; m.albedo = Color(v[0], v[1], v[2]); }
        } else if (type == "emissive") {
            if (!nums(t, 3, 3, v)) return false;
            m.type = MAT_EMISSIVE; m.albedo = Color(0.f, 0.f, 0.f); m.emission = Color(v[0], v[1], v[2]);
        } else return bad("material type must be diffuse, mirror, glass or emissive");
        sc.materials.push_back(m);
        sc.material_names.push_back(name);
        return true;
    }

    bool parse_mesh(const std::vector<std::string>& t) {
        if (t.size() < 2) return bad("mesh <file.obj> [options]");
        std::string file = is_absolute(t[1]) ? t[1] : base_dir + t[1];
        int mat = 0;
        float scale = 1.f, rot_y = 0.f, v[3];
        Vec3 tr(0.f, 0.f, 0.f);
        bool smooth = false;
        for (size_t i = 2; i < t.size();) {
            const std::string& k = t[i++];
            if (k == "material") {
                if (i >= t.size()) return bad("'material' needs a name");
                mat = find_material(t[i]);
                if (mat < 0) return bad("unknown material '" + t[i] + "' (define it before the mesh line)");
                ++i;
            } else if (k == "scale")     { if (!nums(t, i, 1, v)) return false; scale = v[0]; i += 1; }
            else if (k == "rotate_y")    { if (!nums(t, i, 1, v)) return false; rot_y = v[0]; i += 1; }
            else if (k == "translate")   { if (!nums(t, i, 3, v)) return false; tr = Vec3(v[0], v[1], v[2]); i += 3; }
            else if (k == "smooth")      { smooth = true; }
            else return bad("unknown mesh option '" + k + "'");
        }
        if (scale <= 0.f) return bad("scale must be > 0");
        return load_obj(file, mat, scale, rot_y, tr, smooth);
    }

    struct Face { int v[3]; int n[3]; int mat; };

    bool resolve(int idx, size_t count, int& out) {
        if (idx > 0) out = idx - 1;
        else if (idx < 0) out = (int)count + idx;
        else return false;
        return out >= 0 && (size_t)out < count;
    }

    bool load_obj(const std::string& path, int mesh_mat, float scale, float rot_y_deg,
                  const Vec3& tr, bool smooth) {
        std::ifstream in(path);
        if (!in) return bad("cannot open OBJ file '" + path + "'");
        const std::string saved_file = cur_file;
        const int saved_line = line_no;
        cur_file = path;
        line_no = 0;

        float rad = rot_y_deg * kPi / 180.f, cs = cosf(rad), sn = sinf(rad);
        std::vector<Vec3> P, N;
        std::vector<Face> faces;
        int cur_mat = mesh_mat;
        std::string line;
        while (std::getline(in, line)) {
            ++line_no;
            auto t = split_tokens(line);
            if (t.empty()) continue;
            if (t[0] == "v" || t[0] == "vn") {
                float v[3];
                if (t.size() < 4) return bad("expected 3 coordinates");
                for (int k = 0; k < 3; ++k) if (!parse_float(t[1 + k], v[k])) return bad("not a number: " + t[1 + k]);
                if (t[0] == "v") {
                    float x = v[0] * scale, y = v[1] * scale, z = v[2] * scale;
                    P.push_back(Vec3(cs * x + sn * z + tr.x, y + tr.y, -sn * x + cs * z + tr.z));
                } else {
                    N.push_back(normalize(Vec3(cs * v[0] + sn * v[2], v[1], -sn * v[0] + cs * v[2])));
                }
            } else if (t[0] == "usemtl") {
                int m = t.size() > 1 ? find_material(t[1]) : -1;
                if (m < 0) {
                    std::cerr << "warning: " << path << ":" << line_no << ": unknown material '"
                              << (t.size() > 1 ? t[1] : "") << "', using mesh default\n";
                    m = mesh_mat;
                }
                cur_mat = m;
            } else if (t[0] == "f") {
                if (t.size() < 4) return bad("face needs at least 3 vertices");
                std::vector<int> vi, ni;
                for (size_t k = 1; k < t.size(); ++k) {
                    int a = 0, b = 0, c = 0;
                    const std::string& s = t[k];
                    size_t s1 = s.find('/');
                    a = atoi(s.substr(0, s1).c_str());
                    if (s1 != std::string::npos) {
                        size_t s2 = s.find('/', s1 + 1);
                        if (s2 != std::string::npos) c = atoi(s.substr(s2 + 1).c_str());
                    }
                    (void)b;
                    int vr, nr = -1;
                    if (!resolve(a, P.size(), vr)) return bad("vertex index out of range in '" + s + "'");
                    if (c != 0 && !resolve(c, N.size(), nr)) return bad("normal index out of range in '" + s + "'");
                    vi.push_back(vr); ni.push_back(c != 0 ? nr : -1);
                }
                for (size_t k = 1; k + 1 < vi.size(); ++k) {   // fan triangulation
                    Face f;
                    f.v[0] = vi[0]; f.v[1] = vi[k]; f.v[2] = vi[k + 1];
                    f.n[0] = ni[0]; f.n[1] = ni[k]; f.n[2] = ni[k + 1];
                    f.mat = cur_mat;
                    faces.push_back(f);
                }
            }
            // everything else (vt, o, g, s, mtllib, ...) is ignored
        }

        // Optional smooth normals for faces that have none in the file.
        std::vector<Vec3> sn_acc;
        if (smooth) {
            sn_acc.assign(P.size(), Vec3(0.f, 0.f, 0.f));
            for (const Face& f : faces) {
                if (f.n[0] >= 0 && f.n[1] >= 0 && f.n[2] >= 0) continue;
                Vec3 fn = cross(P[f.v[1]] - P[f.v[0]], P[f.v[2]] - P[f.v[0]]);   // area weighted
                for (int k = 0; k < 3; ++k) sn_acc[f.v[k]] += fn;
            }
        }

        for (const Face& f : faces) {
            Triangle tri;
            tri.v0 = P[f.v[0]]; tri.v1 = P[f.v[1]]; tri.v2 = P[f.v[2]];
            Vec3 cr = cross(tri.v1 - tri.v0, tri.v2 - tri.v0);
            if (length(cr) < 1e-12f) { sc.degenerate_skipped++; continue; }
            Vec3 fn = normalize(cr);
            Vec3* out[3] = {&tri.n0, &tri.n1, &tri.n2};
            for (int k = 0; k < 3; ++k) {
                if (f.n[k] >= 0) *out[k] = N[f.n[k]];
                else if (smooth) { Vec3 s = normalize(sn_acc[f.v[k]]); *out[k] = length2(s) > 0.f ? s : fn; }
                else *out[k] = fn;
            }
            tri.material_id = (uint32_t)f.mat;
            sc.triangles.push_back(tri);
        }

        cur_file = saved_file;
        line_no = saved_line;
        return true;
    }

    void finish() {
        sc.camera = make_camera(sc.cam_eye, sc.cam_target, sc.cam_up, sc.cam_fov,
                                (float)sc.settings.width / (float)sc.settings.height);
        float total = 0.f;
        for (size_t i = 0; i < sc.triangles.size(); ++i) {
            const Triangle& tri = sc.triangles[i];
            sc.bounds.expand(tri.bounds());
            const Material& m = sc.materials[tri.material_id];
            if (m.type == MAT_EMISSIVE) {
                Light l;
                l.tri_index = (uint32_t)i;
                l.area = tri.area();
                l.power = l.area * luminance(m.emission);
                total += l.power;
                sc.lights.push_back(l);
            }
        }
        float run = 0.f;
        for (Light& l : sc.lights) { run += l.power; l.cdf = total > 0.f ? run / total : 0.f; }
        if (!sc.lights.empty()) sc.lights.back().cdf = 1.f;
        sc.total_light_power = total;
    }
};

}  // namespace

bool load_scene(const std::string& scene_path, Scene& out, std::string* error) {
    Parser p(out, error);
    return p.parse(scene_path);
}

std::vector<float> Scene::flat_positions() const {
    std::vector<float> v;
    v.reserve(triangles.size() * 9);
    for (const Triangle& t : triangles) {
        const Vec3* p[3] = {&t.v0, &t.v1, &t.v2};
        for (int k = 0; k < 3; ++k) { v.push_back(p[k]->x); v.push_back(p[k]->y); v.push_back(p[k]->z); }
    }
    return v;
}

std::vector<uint32_t> Scene::flat_material_ids() const {
    std::vector<uint32_t> ids(triangles.size());
    for (size_t i = 0; i < triangles.size(); ++i) ids[i] = triangles[i].material_id;
    return ids;
}

}  // namespace rt
