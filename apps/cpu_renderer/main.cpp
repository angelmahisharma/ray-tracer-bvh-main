#include "rt/scene.h"
#include "rt/image.h"
#include "rt/rng.h"
#include "rt/bvh.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#ifdef RT_HAS_OIDN
#include <OpenImageDenoise/oidn.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif

using namespace rt;

#ifdef RT_HAS_OIDN
bool denoise_image(const Image& input, const Image& albedo, const Image& normal,
                   Image& output, std::string& error) {
    OIDNDevice device = oidnNewDevice(OIDN_DEVICE_TYPE_CPU);
    if (!device) { error = "could not create OIDN CPU device"; return false; }
    oidnCommitDevice(device);
    OIDNFilter filter = oidnNewFilter(device, "RT");
    if (!filter) {
        const char* message = nullptr;
        oidnGetDeviceError(device, &message);
        error = message ? message : "could not create OIDN RT filter";
        oidnReleaseDevice(device);
        return false;
    }
    output = Image(input.width, input.height);
    oidnSetSharedFilterImage(filter, "color", const_cast<float*>(input.data()), OIDN_FORMAT_FLOAT3,
                             input.width, input.height, 0, 0, 0);
    oidnSetSharedFilterImage(filter, "output", output.data(), OIDN_FORMAT_FLOAT3,
                             output.width, output.height, 0, 0, 0);
    // OIDN expects albedo in [0, 1] and normals in world/view space in [-1, 1].
    oidnSetSharedFilterImage(filter, "albedo", const_cast<float*>(albedo.data()), OIDN_FORMAT_FLOAT3,
                             input.width, input.height, 0, 0, 0);
    oidnSetSharedFilterImage(filter, "normal", const_cast<float*>(normal.data()), OIDN_FORMAT_FLOAT3,
                             input.width, input.height, 0, 0, 0);
    oidnSetFilterBool(filter, "hdr", true);
    oidnCommitFilter(filter);
    oidnExecuteFilter(filter);
    const char* message = nullptr;
    const OIDNError code = oidnGetDeviceError(device, &message);
    if (code != OIDN_ERROR_NONE) error = message ? message : "OIDN denoising failed";
    oidnReleaseFilter(filter);
    oidnReleaseDevice(device);
    if (code != OIDN_ERROR_NONE) {
        return false;
    }
    return true;
}
#endif

struct FirstHitGuides {
    Color albedo{0.f};
    Vec3 normal{0.f};
};

struct Hit {
    float t = kBigFloat;
    float u = 0.f, v = 0.f;
    uint32_t tri_index = 0;
    Vec3 p, normal;
    bool front_face = true;
};

bool closest_hit(const Scene& scene, const Ray& ray, Hit& hit, const BVH* bvh=nullptr) {
    bool found = false; float closest = ray.tmax;
    auto test = [&](uint32_t i, float& best) {
        const Triangle& tri = scene.triangles[i];
        Vec3 e1 = tri.v1 - tri.v0, e2 = tri.v2 - tri.v0;
        Vec3 pvec = cross(ray.dir, e2);
        float det = dot(e1, pvec);
        if (fabsf(det) < 1e-8f) return false;
        float inv_det = 1.f / det;
        Vec3 tvec = ray.origin - tri.v0;
        float u = dot(tvec, pvec) * inv_det;
        if (u < 0.f || u > 1.f) return false;
        Vec3 qvec = cross(tvec, e1);
        float v = dot(ray.dir, qvec) * inv_det;
        if (v < 0.f || u + v > 1.f) return false;
        float t = dot(e2, qvec) * inv_det;
        if (t <= ray.tmin || t >= best) return false;
        best = t;
        hit.t = t; hit.u = u; hit.v = v; hit.tri_index = i;
        hit.p = ray.at(t);
        hit.front_face = dot(ray.dir, tri.geometric_normal()) < 0.f;
        Vec3 n = tri.shading_normal(u, v);
        if (dot(n, ray.dir) > 0.f) n = -n;
        hit.normal = n;
        return true;
    };
    if (bvh) return bvh->traverse(ray, closest, test, hit.tri_index);
    for (uint32_t i = 0; i < scene.triangles.size(); ++i) found = test(i, closest) || found;
    return found;
}

float power_heuristic(float a, float b) {
    float aa = a * a, bb = b * b;
    return aa / fmaxf(aa + bb, 1e-20f);
}

Vec3 random_in_unit_sphere(Rng& rng) {
    for (int i = 0; i < 100; ++i) {
        Vec3 p(2.f * rng.next_float() - 1.f, 2.f * rng.next_float() - 1.f, 2.f * rng.next_float() - 1.f);
        if (length2(p) < 1.f) return p;
    }
    return Vec3(0.f, 0.f, 0.f);
}

Vec3 random_unit_vector(Rng& rng) { return normalize(random_in_unit_sphere(rng)); }

float schlick(float cosine, float ref_idx) {
    float r0 = (1.f - ref_idx) / (1.f + ref_idx);
    r0 *= r0;
    return r0 + (1.f - r0) * powf(1.f - clampf(cosine, 0.f, 1.f), 5.f);
}

bool sample_light(const Scene& scene, const Hit& surface, Rng& rng, Color& contribution, const BVH* bvh) {
    contribution = Color(0.f);
    if (scene.lights.empty() || scene.total_light_power <= 0.f) return false;

    float pick = rng.next_float();
    const Light* light = &scene.lights.back();
    for (const Light& candidate : scene.lights) {
        if (pick <= candidate.cdf) { light = &candidate; break; }
    }
    if (light->tri_index >= scene.triangles.size() || light->area <= 0.f || light->power <= 0.f) return false;

    const Triangle& light_tri = scene.triangles[light->tri_index];
    const Material& light_mat = scene.materials[light_tri.material_id];
    float root = sqrtf(rng.next_float());
    float r2 = rng.next_float();
    float b0 = 1.f - root;
    float b1 = root * (1.f - r2);
    float b2 = root * r2;
    Vec3 light_point = light_tri.v0 * b0 + light_tri.v1 * b1 + light_tri.v2 * b2;
    Vec3 delta = light_point - surface.p;
    float dist2 = length2(delta);
    if (dist2 <= kRayEps * kRayEps) return false;
    float dist = sqrtf(dist2);
    Vec3 wi = delta / dist;
    float cos_surface = fmaxf(0.f, dot(surface.normal, wi));
    Vec3 light_normal = light_tri.geometric_normal();
    float cos_light = dot(light_normal, -wi);
    if (cos_surface <= 0.f || cos_light <= 0.f) return false;

    Ray shadow(surface.p + surface.normal * kRayEps, wi, kRayEps, dist - kRayEps);
    Hit blocker;
    if (shadow.tmax <= shadow.tmin || closest_hit(scene, shadow, blocker, bvh)) return false;

    float select_pdf = light->power / scene.total_light_power;
    float light_pdf = select_pdf * dist2 / (light->area * cos_light);
    if (light_pdf <= 0.f) return false;
    float bsdf_pdf = cos_surface / kPi;
    float mis = power_heuristic(light_pdf, bsdf_pdf);
    Color bsdf = light_mat.emission * (cos_surface / (kPi * light_pdf));
    // Diffuse BRDF albedo is applied by the caller.
    contribution = bsdf * mis;
    return true;
}

float light_pdf_for_hit(const Scene& scene, uint32_t tri_index, const Vec3& from, const Vec3& hit_point, const Vec3& ray_dir) {
    for (const Light& light : scene.lights) {
        if (light.tri_index != tri_index || light.area <= 0.f || scene.total_light_power <= 0.f) continue;
        const Triangle& tri = scene.triangles[tri_index];
        float cos_light = dot(tri.geometric_normal(), -ray_dir);
        if (cos_light <= 0.f) return 0.f;
        float dist2 = length2(hit_point - from);
        float select_pdf = light.power / scene.total_light_power;
        return select_pdf * dist2 / (light.area * cos_light);
    }
    return 0.f;
}

Color trace(const Scene& scene, Ray ray, Rng& rng, const BVH* bvh, FirstHitGuides* guides) {
    Color throughput(1.f), radiance(0.f);
    bool previous_was_diffuse = false;
    float previous_bsdf_pdf = 0.f;
    Vec3 previous_point(0.f);

    for (int bounce = 0; bounce <= scene.settings.max_bounces; ++bounce) {
        Hit hit;
        if (!closest_hit(scene, ray, hit, bvh)) {
            radiance += throughput * scene.env.eval(normalize(ray.dir));
            break;
        }
        const Triangle& tri = scene.triangles[hit.tri_index];
        if (tri.material_id >= scene.materials.size()) break;
        const Material& mat = scene.materials[tri.material_id];

        if (bounce == 0 && guides) {
            // Match the first-hit guide convention: matte color for diffuse,
            // reflectance for mirrors, neutral albedo for dielectric glass.
            if (mat.type == MAT_DIFFUSE || mat.type == MAT_MIRROR) {
                guides->albedo = Color(clampf(mat.albedo.x, 0.f, 1.f),
                                       clampf(mat.albedo.y, 0.f, 1.f),
                                       clampf(mat.albedo.z, 0.f, 1.f));
            } else if (mat.type == MAT_GLASS) {
                guides->albedo = Color(1.f);
            }
            guides->normal = hit.normal;
        }

        if (mat.type == MAT_EMISSIVE) {
            float weight = 1.f;
            if (previous_was_diffuse) {
                float light_pdf = light_pdf_for_hit(scene, hit.tri_index, previous_point, hit.p, normalize(ray.dir));
                weight = power_heuristic(previous_bsdf_pdf, light_pdf);
            }
            // Emissive triangles are treated as one-sided, matching their winding.
            if (hit.front_face || bounce == 0)
                radiance += throughput * mat.emission * weight;
            break;
        }
        if (bounce == scene.settings.max_bounces) break;

        Vec3 next_dir;
        previous_was_diffuse = false;
        if (mat.type == MAT_DIFFUSE) {
            Color direct(0.f);
            if (sample_light(scene, hit, rng, direct, bvh)) radiance += throughput * (mat.albedo * direct);
            next_dir = normalize(hit.normal + random_unit_vector(rng));
            if (length2(next_dir) < 1e-12f) next_dir = hit.normal;
            previous_bsdf_pdf = fmaxf(0.f, dot(hit.normal, next_dir)) / kPi;
            previous_point = hit.p;
            previous_was_diffuse = true;
            throughput = throughput * mat.albedo;
        } else if (mat.type == MAT_MIRROR) {
            next_dir = normalize(reflect(normalize(ray.dir), hit.normal) + mat.fuzz * random_in_unit_sphere(rng));
            if (dot(next_dir, hit.normal) <= 0.f) break;
            throughput = throughput * mat.albedo;
        } else if (mat.type == MAT_GLASS) {
            float ratio = hit.front_face ? 1.f / mat.ior : mat.ior;
            Vec3 unit_dir = normalize(ray.dir);
            float cos_theta = fminf(dot(-unit_dir, hit.normal), 1.f);
            float sin_theta = sqrtf(fmaxf(0.f, 1.f - cos_theta * cos_theta));
            bool cannot_refract = ratio * sin_theta > 1.f;
            if (cannot_refract || schlick(cos_theta, mat.ior) > rng.next_float()) {
                next_dir = reflect(unit_dir, hit.normal);
            } else {
                Vec3 perp = (unit_dir + hit.normal * cos_theta) * ratio;
                Vec3 parallel = hit.normal * (-sqrtf(fmaxf(0.f, 1.f - length2(perp))));
                next_dir = perp + parallel;
            }
            throughput = throughput * mat.albedo;
        } else {
            break;
        }

        float side = dot(next_dir, hit.normal) >= 0.f ? 1.f : -1.f;
        ray = Ray(hit.p + hit.normal * (side * kRayEps), normalize(next_dir), kRayEps);
    }
    return radiance;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: cpu_renderer <scene.scene> [output_prefix] [spp] [max_bounces] [brute|median|sah] [--denoise] [--threads N]\n";
        return 1;
    }
    Scene scene;
    std::string error;
    if (!load_scene(argv[1], scene, &error)) {
        std::cerr << "Scene load failed: " << error << '\n';
        return 1;
    }
    if (argc >= 4) scene.settings.spp = std::max(1, std::atoi(argv[3]));
    if (argc >= 5) scene.settings.max_bounces = std::max(0, std::atoi(argv[4]));
    std::string prefix = argc >= 3 ? argv[2] : "output";
    const std::string mode = argc >= 6 ? argv[5] : "brute";
    if (mode != "brute" && mode != "median" && mode != "sah") { std::cerr << "Unknown mode: " << mode << '\n'; return 1; }
    bool denoise = false;
    int requested_threads = 0;
    for (int i = 6; i < argc; ++i) {
        if (std::string(argv[i]) == "--denoise") denoise = true;
        else if (std::string(argv[i]) == "--threads" && i + 1 < argc) {
            requested_threads = std::atoi(argv[++i]);
            if (requested_threads < 1) {
                std::cerr << "--threads requires a positive integer.\n";
                return 1;
            }
        }
        else { std::cerr << "Unknown option: " << argv[i] << '\n'; return 1; }
    }
#ifdef _OPENMP
    if (requested_threads > 0) {
        omp_set_dynamic(0);
        omp_set_num_threads(requested_threads);
    }
#else
    if (requested_threads > 0) {
        std::cerr << "This build has no OpenMP support; rebuild with OpenMP enabled before using --threads.\n";
        return 1;
    }
#endif
#ifndef RT_HAS_OIDN
    if (denoise) {
        std::cerr << "Denoising was requested, but this build has no OIDN support. Configure with RT_ENABLE_OIDN=ON and set OpenImageDenoise_DIR.\n";
        return 1;
    }
#endif
    BVH bvh; const BVH* accel = nullptr; double build_seconds = 0.0;
    if (mode != "brute") {
        const auto build_start = std::chrono::steady_clock::now();
        bvh.build(scene, mode == "sah" ? BVHStrategy::SAH : BVHStrategy::Median);
        build_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - build_start).count(); accel = &bvh;
        std::cout << "BVH nodes: " << bvh.stats.nodes << " internal: " << bvh.stats.internal << " leaves: " << bvh.stats.leaves
                  << " depth: " << bvh.stats.max_depth << " avg triangles/leaf: " << bvh.stats.average_triangles_per_leaf()
                  << " max leaf: " << bvh.stats.max_leaf << " references: " << bvh.stats.references << "\n";
    }
    const int w = scene.settings.width, h = scene.settings.height;
    Image img(w, h);
#ifdef RT_HAS_OIDN
    Image albedo_guide, normal_guide;
    if (denoise) {
        albedo_guide = Image(w, h);
        normal_guide = Image(w, h);
    }
#endif
    const auto start = std::chrono::steady_clock::now();

    #pragma omp parallel for schedule(dynamic, 1)
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            Rng rng;
            rng.seed(1234, (uint64_t)y * (uint64_t)w + (uint64_t)x);
            Color sum(0.f);
#ifdef RT_HAS_OIDN
            Color albedo_sum(0.f), normal_sum(0.f);
#endif
            for (int s = 0; s < scene.settings.spp; ++s) {
                Ray ray = pixel_ray(scene.camera, x, y, w, h, rng.next_float(), rng.next_float());
                FirstHitGuides guides;
                sum += trace(scene, ray, rng, accel,
#ifdef RT_HAS_OIDN
                             denoise ? &guides : nullptr
#else
                             nullptr
#endif
                );
#ifdef RT_HAS_OIDN
                if (denoise) {
                    albedo_sum += guides.albedo;
                    normal_sum += guides.normal;
                }
#endif
            }
            img.set(x, y, sum / (float)scene.settings.spp);
#ifdef RT_HAS_OIDN
            if (denoise) {
                albedo_guide.set(x, y, albedo_sum / (float)scene.settings.spp);
                normal_guide.set(x, y, normal_sum / (float)scene.settings.spp);
            }
#endif
        }
    }

    const auto end = std::chrono::steady_clock::now();
    const double seconds = std::chrono::duration<double>(end - start).count();
#ifdef RT_HAS_OIDN
    if (denoise) {
        Image denoised;
        std::string denoise_error;
        const auto denoise_start = std::chrono::steady_clock::now();
        if (!denoise_image(img, albedo_guide, normal_guide, denoised, denoise_error)) {
            std::cerr << "OIDN failed: " << denoise_error << '\n';
            return 1;
        }
        const double denoise_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - denoise_start).count();
        const std::string denoised_prefix = prefix + "_denoised";
        if (!save_pfm(denoised, denoised_prefix + ".pfm") || !save_png(denoised, denoised_prefix + ".png")) {
            std::cerr << "Failed to save denoised output image(s).\n";
            return 1;
        }
        std::cout << "Denoise time: " << denoise_seconds << " s\nSaved " << denoised_prefix << ".png and " << denoised_prefix << ".pfm\n";
    }
#endif
    const bool pfm_ok = save_pfm(img, prefix + ".pfm");
    const bool png_ok = save_png(img, prefix + ".png");
    std::cout << "Triangles: " << scene.triangles.size() << "\nResolution: " << w << "x" << h
              << "\nMode: " << mode << "\nBVH build time: " << (accel ? std::to_string(build_seconds) + " s" : "n/a")
              << "\nSPP: " << scene.settings.spp << "\nMax bounces: " << scene.settings.max_bounces
              << "\nRender time: " << seconds << " s\n";
#ifdef _OPENMP
    std::cout << "OpenMP threads: " << omp_get_max_threads()
              << (requested_threads > 0 ? " (requested)\n" : " (available)\n");
#else
    std::cout << "OpenMP: disabled (single-threaded build)\n";
#endif
    if (!pfm_ok || !png_ok) {
        std::cerr << "Failed to save output image(s).\n";
        return 1;
    }
    std::cout << "Saved " << prefix << ".png and " << prefix << ".pfm\n";
    return 0;
}
