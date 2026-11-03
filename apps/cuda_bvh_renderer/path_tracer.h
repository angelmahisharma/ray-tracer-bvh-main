// Path tracer used by the CUDA kernel. Line-for-line the same algorithm as
// apps/cpu_renderer/main.cpp (diffuse NEE + MIS, mirror, glass, emissive, sky)
// and the same RNG draw order, so the images match up to float rounding.
// Plain RT_HD code: also runs on the CPU for tests.
#pragma once
#include "scene_view.h"
#include "rt/rng.h"

namespace rt {

struct Hit {
    float t = kBigFloat, u = 0.f, v = 0.f;
    uint32_t tri_index = 0;
    Vec3 p, normal;
    bool front_face = true;
};

RT_HD inline bool brute_closest(const SceneView& sc, const Ray& ray, BvhHit& h) {
    bool found = false;
    float closest = ray.tmax;
    for (uint32_t i = 0; i < sc.num_tris; ++i) {
        float t, u, v;
        if (intersect_triangle(sc.tris[i], ray, closest, t, u, v)) {
            closest = t; h.t = t; h.u = u; h.v = v; h.tri = i; found = true;
        }
    }
    return found;
}

RT_HD inline bool closest_hit(const SceneView& sc, const Ray& ray, Hit& hit) {
    BvhHit h;
    bool found = sc.use_bvh ? bvh_closest_hit(sc.nodes, sc.tris, ray, h) : brute_closest(sc, ray, h);
    if (!found) return false;
    const Triangle& tri = sc.tris[h.tri];
    hit.t = h.t; hit.u = h.u; hit.v = h.v; hit.tri_index = h.tri;
    hit.p = ray.at(h.t);
    hit.front_face = dot(ray.dir, tri.geometric_normal()) < 0.f;
    Vec3 n = tri.shading_normal(h.u, h.v);
    if (dot(n, ray.dir) > 0.f) n = -n;
    hit.normal = n;
    return true;
}

RT_HD inline bool occluded(const SceneView& sc, const Ray& ray) {
    if (sc.use_bvh) return bvh_any_hit(sc.nodes, sc.tris, ray);
    BvhHit h;
    return brute_closest(sc, ray, h);
}

RT_HD inline float power_heuristic(float a, float b) {
    float aa = a * a, bb = b * b;
    return aa / fmaxf(aa + bb, 1e-20f);
}

RT_HD inline Vec3 random_in_unit_sphere(Rng& rng) {
    for (int i = 0; i < 100; ++i) {
        // C++ leaves the order of function arguments unspecified (g++ evaluates right to left, nvcc left to
        // right), so the draw order is spelled out. z, y, x is what cpu_renderer built with g++ does.
        float pz = 2.f * rng.next_float() - 1.f;
        float py = 2.f * rng.next_float() - 1.f;
        float px = 2.f * rng.next_float() - 1.f;
        Vec3 p(px, py, pz);
        if (length2(p) < 1.f) return p;
    }
    return Vec3(0.f, 0.f, 0.f);
}

RT_HD inline Vec3 random_unit_vector(Rng& rng) { return normalize(random_in_unit_sphere(rng)); }

RT_HD inline float schlick(float cosine, float ref_idx) {
    float r0 = (1.f - ref_idx) / (1.f + ref_idx);
    r0 *= r0;
    return r0 + (1.f - r0) * powf(1.f - clampf(cosine, 0.f, 1.f), 5.f);
}

// First light whose cdf >= pick (binary search; same choice as the CPU linear scan).
RT_HD inline const Light* pick_light(const SceneView& sc, float pick) {
    uint32_t lo = 0, hi = sc.num_lights - 1;
    while (lo < hi) {
        uint32_t mid = (lo + hi) / 2;
        if (pick <= sc.lights[mid].cdf) hi = mid; else lo = mid + 1;
    }
    return &sc.lights[lo];
}

RT_HD inline bool sample_light(const SceneView& sc, const Hit& surface, Rng& rng, Color& contribution) {
    contribution = Color(0.f);
    if (sc.num_lights == 0 || sc.total_light_power <= 0.f) return false;

    float pick = rng.next_float();
    const Light* light = pick_light(sc, pick);
    if (light->tri_index >= sc.num_tris || light->area <= 0.f || light->power <= 0.f) return false;

    const Triangle& light_tri = sc.tris[light->tri_index];
    const Material& light_mat = sc.materials[light_tri.material_id];
    float root = sqrtf(rng.next_float());
    float r2 = rng.next_float();
    float b0 = 1.f - root, b1 = root * (1.f - r2), b2 = root * r2;
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
    if (shadow.tmax <= shadow.tmin || occluded(sc, shadow)) return false;

    float select_pdf = light->power / sc.total_light_power;
    float light_pdf = select_pdf * dist2 / (light->area * cos_light);
    if (light_pdf <= 0.f) return false;
    float bsdf_pdf = cos_surface / kPi;
    float mis = power_heuristic(light_pdf, bsdf_pdf);
    Color bsdf = light_mat.emission * (cos_surface / (kPi * light_pdf));
    contribution = bsdf * mis;   // diffuse albedo is applied by the caller
    return true;
}

RT_HD inline float light_pdf_for_hit(const SceneView& sc, uint32_t tri_index, const Vec3& from,
                                     const Vec3& hit_point, const Vec3& ray_dir) {
    int32_t li = sc.tri_light[tri_index];
    if (li < 0) return 0.f;
    const Light& light = sc.lights[li];
    if (light.area <= 0.f || sc.total_light_power <= 0.f) return 0.f;
    float cos_light = dot(sc.tris[tri_index].geometric_normal(), -ray_dir);
    if (cos_light <= 0.f) return 0.f;
    float dist2 = length2(hit_point - from);
    return (light.power / sc.total_light_power) * dist2 / (light.area * cos_light);
}

RT_HD inline Color trace(const SceneView& sc, Ray ray, Rng& rng) {
    Color throughput(1.f), radiance(0.f);
    bool previous_was_diffuse = false;
    float previous_bsdf_pdf = 0.f;
    Vec3 previous_point(0.f);

    for (int bounce = 0; bounce <= sc.max_bounces; ++bounce) {
        Hit hit;
        if (!closest_hit(sc, ray, hit)) {
            radiance += throughput * sc.env.eval(normalize(ray.dir));
            break;
        }
        const Triangle& tri = sc.tris[hit.tri_index];
        if (tri.material_id >= sc.num_materials) break;
        const Material& mat = sc.materials[tri.material_id];

        if (mat.type == MAT_EMISSIVE) {
            float weight = 1.f;
            if (previous_was_diffuse) {
                float light_pdf = light_pdf_for_hit(sc, hit.tri_index, previous_point, hit.p, normalize(ray.dir));
                weight = power_heuristic(previous_bsdf_pdf, light_pdf);
            }
            if (hit.front_face || bounce == 0) radiance += throughput * mat.emission * weight;
            break;
        }
        if (bounce == sc.max_bounces) break;

        Vec3 next_dir;
        previous_was_diffuse = false;
        if (mat.type == MAT_DIFFUSE) {
            Color direct(0.f);
            if (sample_light(sc, hit, rng, direct)) radiance += throughput * (mat.albedo * direct);
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

// One pixel, all samples. Same seeding as cpu_renderer: seed(global_seed, y*w+x).
RT_HD inline Color render_pixel(const SceneView& sc, int x, int y) {
    Rng rng;
    rng.seed(sc.seed, (uint64_t)y * (uint64_t)sc.width + (uint64_t)x);
    Color sum(0.f);
    for (int s = 0; s < sc.spp; ++s) {
        float jy = rng.next_float();   // g++ evaluates pixel_ray's arguments right to left (as cpu_renderer does)
        float jx = rng.next_float();
        Ray ray = pixel_ray(sc.camera, x, y, sc.width, sc.height, jx, jy);
        sum += trace(sc, ray, rng);
    }
    return sum / (float)sc.spp;
}

}  // namespace rt
