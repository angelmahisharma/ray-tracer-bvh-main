#include <optix.h>
#include "launch_params.h"
#include "rt/camera.h"
#include "rt/material.h"
#include "rt/rng.h"

// RT_MATCH_CPU=1: consume random numbers exactly like cpu_renderer (same
// rejection loops, same draw order) so the GPU image can be diffed pixel by
// pixel against the CPU one. Default: closed-form sampling with the same
// distributions and no data-dependent loops.
#ifndef RT_MATCH_CPU
#define RT_MATCH_CPU 0
#endif

using namespace rt;

extern "C" {
__constant__ LaunchParams params;
}

namespace {

constexpr unsigned kRadianceFlags = OPTIX_RAY_FLAG_DISABLE_ANYHIT;
constexpr unsigned kShadowFlags = OPTIX_RAY_FLAG_DISABLE_ANYHIT | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT |
                                  OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT;

__forceinline__ __device__ Vec3 v3(const float3& f) { return Vec3(f.x, f.y, f.z); }
__forceinline__ __device__ Vec3 v3(const float4& f) { return Vec3(f.x, f.y, f.z); }
__forceinline__ __device__ float3 f3(const Vec3& v) { return make_float3(v.x, v.y, v.z); }

__forceinline__ __device__ uint32_t pixel_index() {
    const uint3 i = optixGetLaunchIndex();
    return i.y * params.width + i.x;
}

// rt::Rng with only `state` carried around: `inc` is the stream set by
// rng.seed(seed, pixel_index), so it is rebuilt from the launch index.
struct PixelRng {
    uint64_t state;

    __forceinline__ __device__ float next_float() {
        Rng r;
        r.state = state;
        r.inc = ((uint64_t)pixel_index() << 1u) | 1u;
        float f = r.next_float();
        state = r.state;
        return f;
    }
};

// Casts a ray; the result is left in the outgoing hit object.
__forceinline__ __device__ void traverse(const Vec3& origin, const Vec3& dir, float tmin, float tmax,
                                         unsigned flags) {
    optixTraverse(params.handle, f3(origin), f3(dir), tmin, tmax, 0.f, OptixVisibilityMask(255), flags,
                  0, 1, 0);
}

#if RT_MATCH_CPU
// MSVC evaluates function / constructor arguments right to left, so the CPU
// renderer's `Vec3(2*r()-1, 2*r()-1, 2*r()-1)` fills z first, and
// `pixel_ray(..., r(), r())` draws jy before jx.
__device__ Vec3 random_in_unit_sphere(PixelRng& rng) {
    for (int i = 0; i < 100; ++i) {
        float z = 2.f * rng.next_float() - 1.f;
        float y = 2.f * rng.next_float() - 1.f;
        float x = 2.f * rng.next_float() - 1.f;
        Vec3 p(x, y, z);
        if (length2(p) < 1.f) return p;
    }
    return Vec3(0.f, 0.f, 0.f);
}
__forceinline__ __device__ Vec3 random_unit_vector(PixelRng& rng) { return normalize(random_in_unit_sphere(rng)); }
#else
__forceinline__ __device__ Vec3 random_unit_vector(PixelRng& rng) {
    float z = 1.f - 2.f * rng.next_float();
    float r = sqrtf(fmaxf(0.f, 1.f - z * z));
    float s, c;
    sincosf(2.f * kPi * rng.next_float(), &s, &c);
    return Vec3(r * c, r * s, z);
}
__forceinline__ __device__ Vec3 random_in_unit_sphere(PixelRng& rng) {
    Vec3 d = random_unit_vector(rng);
    return d * cbrtf(rng.next_float());
}
#endif

__forceinline__ __device__ Ray camera_ray(float jx, float jy) {
    Camera cam;
    cam.origin = v3(params.cam_origin);
    cam.lower_left = v3(params.cam_lower_left);
    cam.horizontal = v3(params.cam_horizontal);
    cam.vertical = v3(params.cam_vertical);
    const uint3 i = optixGetLaunchIndex();
    return pixel_ray(cam, (int)i.x, (int)i.y, (int)params.width, (int)params.height, jx, jy);
}

__forceinline__ __device__ Ray camera_ray(PixelRng& rng) {
#if RT_MATCH_CPU
    float jy = rng.next_float();
    float jx = rng.next_float();
#else
    float jx = rng.next_float();
    float jy = rng.next_float();
#endif
    return camera_ray(jx, jy);
}

// Interpolated shading normal of the hit (TriShade from params.tris), facing against `dir`.
__forceinline__ __device__ Vec3 shading_normal(const uint4& tri, const float2& bary, const Vec3& dir) {
    const float u = bary.x, v = bary.y;
    Vec3 n = normalize(oct_decode(tri.x) * (1.f - u - v) + oct_decode(tri.y) * u + oct_decode(tri.z) * v);
    return dot(n, dir) > 0.f ? -n : n;
}

__forceinline__ __device__ Color env_radiance(const Vec3& dir) {
    Environment env;
    env.mode = params.env_mode;
    env.a = v3(params.env_a);
    env.b = v3(params.env_b);
    return env.eval(dir);
}

__forceinline__ __device__ float power_heuristic(float a, float b) {
    float aa = a * a, bb = b * b;
    return aa / fmaxf(aa + bb, 1e-20f);
}

__forceinline__ __device__ float schlick(float cosine, float ref_idx) {
    float r0 = (1.f - ref_idx) / (1.f + ref_idx);
    r0 *= r0;
    float m = 1.f - clampf(cosine, 0.f, 1.f), m2 = m * m;
    return r0 + (1.f - r0) * (m2 * m2 * m);
}

// A shadow ray plus what it adds if the light turns out to be visible.
struct LightSample {
    Vec3 origin, dir;
    float tmax;
    Color contribution;   // MIS-weighted, without albedo and throughput
};

// Next-event estimation, first half: pick a light by power and sample a point
// on it (same math and random draws as cpu_renderer's sample_light). The shadow
// ray is cast later by the caller, once the bounce has been sampled, so less
// state is live across it.
__device__ bool sample_light(const Vec3& p, const Vec3& n, PixelRng& rng, LightSample& out) {
    const uint32_t count = params.num_lights;
    if (count == 0) return false;

    // First light with pick <= cdf (the last one if none), like the CPU's linear scan.
    float pick = rng.next_float();
    uint32_t lo = 0, hi = count - 1;
    while (lo < hi) {
        uint32_t mid = (lo + hi) >> 1;
        if (pick <= __ldg(params.light_cdf + mid)) hi = mid; else lo = mid + 1;
    }
    const DevLight& light = params.lights[lo];
    const float4 v0_area = __ldg(&light.v0_area);
    const float4 e1_pdf = __ldg(&light.e1_pdf);
    const float area = v0_area.w, select_pdf = e1_pdf.w;
    if (area <= 0.f || select_pdf <= 0.f) return false;

    const Vec3 e1 = v3(e1_pdf), e2 = v3(__ldg(&light.e2));
    float root = sqrtf(rng.next_float());
    float r2 = rng.next_float();
    Vec3 light_point = v3(v0_area) + e1 * (root * (1.f - r2)) + e2 * (root * r2);
    Vec3 delta = light_point - p;
    float dist2 = length2(delta);
    if (dist2 <= kRayEps * kRayEps) return false;
    float dist = sqrtf(dist2);
    Vec3 wi = delta / dist;
    float cos_surface = fmaxf(0.f, dot(n, wi));
    float cos_light = dot(normalize(cross(e1, e2)), -wi);
    if (cos_surface <= 0.f || cos_light <= 0.f) return false;

    out.tmax = dist - kRayEps;
    if (out.tmax <= kRayEps) return false;
    float light_pdf = select_pdf * dist2 / (area * cos_light);
    if (light_pdf <= 0.f) return false;
    float bsdf_pdf = cos_surface / kPi;
    float weight = power_heuristic(light_pdf, bsdf_pdf);
    out.origin = p + n * kRayEps;
    out.dir = wi;
    out.contribution = v3(__ldg(&light.emission)) * (cos_surface / (kPi * light_pdf) * weight);
    return true;
}

// Probability that sample_light() would have produced this hit on light triangle `prim`.
__device__ float light_pdf_for_hit(uint32_t prim, const Vec3& from, const Vec3& hit_point, const Vec3& dir) {
    const int32_t index = __ldg(params.tri_to_light + prim);
    if (index < 0) return 0.f;
    const DevLight& light = params.lights[index];
    const float4 v0_area = __ldg(&light.v0_area);
    const float4 e1_pdf = __ldg(&light.e1_pdf);
    if (v0_area.w <= 0.f || e1_pdf.w <= 0.f) return 0.f;
    float cos_light = dot(normalize(cross(v3(e1_pdf), v3(__ldg(&light.e2)))), -dir);
    if (cos_light <= 0.f) return 0.f;
    return e1_pdf.w * length2(hit_point - from) / (v0_area.w * cos_light);
}

}  // namespace

extern "C" __global__ void __raygen__path() {
    PixelRng rng;
    {
        Rng seeded;
        seeded.seed(params.seed, pixel_index());
        rng.state = seeded.state;
    }

    // Path regeneration: one loop iteration = one bounce. When a path ends the
    // thread immediately starts its next sample instead of idling until the
    // longest path in its warp is done. Each pixel still consumes its own
    // random stream in the same order as the CPU renderer.
    Color sum(0.f), throughput(1.f);
    Vec3 prev_point(0.f);
    float prev_bsdf_pdf = -1.f;   // < 0: the previous bounce was not diffuse (no MIS)
    uint32_t bounce = 0, sample = 0, rays = 0;
    Ray ray = camera_ray(rng);

    for (;;) {
        traverse(ray.origin, ray.dir, kRayEps, kBigFloat, kRadianceFlags);
        ++rays;
        // Shader Execution Reordering: regroup threads by hit/miss and material
        // type (= hit-group record) so the shading below runs coherently.
        // Camera-ray hits are already coherent, so reordering them only costs
        // time (measured on spheres: every bounce 71.6 ms, skipping bounce 0
        // 62.6 ms, only from bounce 2 on 96 ms, never 88.8 ms).
        if (params.reorder && bounce > 0) optixReorder();

        const Vec3 dir = v3(optixHitObjectGetWorldRayDirection());
        bool path_done = true;
        if (optixHitObjectIsMiss()) {
            sum += throughput * env_radiance(normalize(dir));
        } else {
            const uint32_t prim = optixHitObjectGetPrimitiveIndex();
            const Vec3 hit_point = v3(optixHitObjectGetWorldRayOrigin()) + dir * optixHitObjectGetRayTmax();
            const bool front_face = optixIsFrontFaceHit(optixHitObjectGetHitKind());
            const uint4 tri = __ldg(reinterpret_cast<const uint4*>(params.tris) + prim);
            const float4 albedo_type = __ldg(&params.materials[tri.w].albedo_type);
            const float4 emission_param = __ldg(&params.materials[tri.w].emission_param);
            const uint32_t type = (uint32_t)albedo_type.w;

            if (type == MAT_EMISSIVE) {
                float weight = 1.f;
                if (prev_bsdf_pdf >= 0.f)
                    weight = power_heuristic(prev_bsdf_pdf,
                                             light_pdf_for_hit(prim, prev_point, hit_point, normalize(dir)));
                // Emissive triangles are one-sided, matching their winding.
                if (front_face || bounce == 0) sum += throughput * v3(emission_param) * weight;
            } else if (bounce < params.max_bounces) {
                const Vec3 n = shading_normal(tri, optixHitObjectGetTriangleBarycentrics(), dir);

                Vec3 next_dir;
                LightSample light;
                bool shadow_ray = false;
                path_done = false;
                prev_bsdf_pdf = -1.f;
                if (type == MAT_DIFFUSE) {
                    shadow_ray = sample_light(hit_point, n, rng, light);
                    next_dir = normalize(n + random_unit_vector(rng));
                    if (length2(next_dir) < 1e-12f) next_dir = n;
                    prev_bsdf_pdf = fmaxf(0.f, dot(n, next_dir)) / kPi;
                    prev_point = hit_point;
                } else if (type == MAT_MIRROR) {
                    const float fuzz = emission_param.w;
                    const Vec3 r = reflect(normalize(dir), n);
#if RT_MATCH_CPU
                    next_dir = normalize(r + fuzz * random_in_unit_sphere(rng));
#else
                    next_dir = fuzz > 0.f ? normalize(r + fuzz * random_in_unit_sphere(rng)) : normalize(r);
#endif
                    if (dot(next_dir, n) <= 0.f) path_done = true;
                } else if (type == MAT_GLASS) {
                    const float ior = emission_param.w;
                    const float ratio = front_face ? 1.f / ior : ior;
                    const Vec3 unit_dir = normalize(dir);
                    const float cos_theta = fminf(dot(-unit_dir, n), 1.f);
                    const float sin_theta = sqrtf(fmaxf(0.f, 1.f - cos_theta * cos_theta));
                    const bool cannot_refract = ratio * sin_theta > 1.f;
                    if (cannot_refract || schlick(cos_theta, ior) > rng.next_float()) {
                        next_dir = reflect(unit_dir, n);
                    } else {
                        Vec3 perp = (unit_dir + n * cos_theta) * ratio;
                        Vec3 parallel = n * (-sqrtf(fmaxf(0.f, 1.f - length2(perp))));
                        next_dir = perp + parallel;
                    }
                } else {
                    path_done = true;
                }

                if (!path_done) {
                    throughput = throughput * v3(albedo_type);
                    const float side = dot(next_dir, n) >= 0.f ? 1.f : -1.f;
                    ray = Ray(hit_point + n * (side * kRayEps), normalize(next_dir), kRayEps);
                    // Second half of next-event estimation (CPU: throughput * (albedo * direct)).
                    if (shadow_ray) {
                        traverse(light.origin, light.dir, kRayEps, light.tmax, kShadowFlags);
                        ++rays;
                        if (optixHitObjectIsMiss()) sum += throughput * light.contribution;
                    }
                }
            }
        }

        if (path_done) {
            if (++sample == params.spp) break;
            ray = camera_ray(rng);
            throughput = Color(1.f);
            prev_bsdf_pdf = -1.f;
            bounce = 0;
        } else {
            ++bounce;
        }
    }

    const float inv_spp = 1.f / (float)params.spp;
    float4 result = make_float4(sum.x * inv_spp, sum.y * inv_spp, sum.z * inv_spp, 1.f);
    const uint32_t pixel = pixel_index();
    if (params.frame_index > 0) {
        // Progressive refinement (viewer): running mean over frames with an unchanged camera.
        const float4 prev = params.accum[pixel];
        const float t = 1.f / (float)(params.frame_index + 1);
        result = make_float4(prev.x + (result.x - prev.x) * t, prev.y + (result.y - prev.y) * t,
                             prev.z + (result.z - prev.z) * t, 1.f);
    }
    params.accum[pixel] = result;
    if (params.ray_counter) atomicAdd(params.ray_counter, (unsigned long long)rays);
}

// Viewer display pass: linear radiance -> 8-bit sRGB (same transfer as the PNG
// output), rows flipped because OpenGL textures start at the bottom row.
extern "C" __global__ void __raygen__display() {
    const uint3 i = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();
    const float4 c = params.display_src[i.y * dim.x + i.x];
    auto to8 = [](float v) { return (unsigned char)(linear_to_srgb(v) * 255.f + 0.5f); };
    params.display[(dim.y - 1 - i.y) * dim.x + i.x] = make_uchar4(to8(c.x), to8(c.y), to8(c.z), 255);
}

// Denoiser guide pass: albedo and world-space normal of the first non-specular
// surface, following perfect mirror/glass chains (fuzz ignored, refraction
// preferred over reflection) so the denoiser also sees edges inside
// reflections. Averaged over a few jittered camera rays for antialiasing.
// It is a separate program so the path tracer carries no extra live state.
extern "C" __global__ void __raygen__guides() {
    constexpr uint32_t kMaxSpecularDepth = 8;
    PixelRng rng;
    {
        Rng seeded;
        seeded.seed(params.seed ^ 0x5bd1e995u, pixel_index());
        rng.state = seeded.state;
    }

    const uint32_t spp = params.guide_spp;
    Color albedo_sum(0.f);
    Vec3 normal_sum(0.f);
    for (uint32_t s = 0; s < spp; ++s) {
        const Ray ray = spp == 1 ? camera_ray(0.5f, 0.5f) : camera_ray(rng);
        Vec3 origin = ray.origin, dir = ray.dir;
        Color tint(1.f), albedo(0.f);
        Vec3 normal(0.f);
        for (uint32_t depth = 0; depth < kMaxSpecularDepth; ++depth) {
            traverse(origin, dir, kRayEps, kBigFloat, kRadianceFlags);
            if (optixHitObjectIsMiss()) {
                const Color sky = env_radiance(dir);
                albedo = tint * Color(clampf(sky.x, 0.f, 1.f), clampf(sky.y, 0.f, 1.f), clampf(sky.z, 0.f, 1.f));
                break;
            }
            const uint32_t prim = optixHitObjectGetPrimitiveIndex();
            const Vec3 hit_point = origin + dir * optixHitObjectGetRayTmax();
            const uint4 tri = __ldg(reinterpret_cast<const uint4*>(params.tris) + prim);
            const float4 albedo_type = __ldg(&params.materials[tri.w].albedo_type);
            const uint32_t type = (uint32_t)albedo_type.w;
            const Vec3 n = shading_normal(tri, optixHitObjectGetTriangleBarycentrics(), dir);

            if (type == MAT_DIFFUSE) {
                albedo = tint * v3(albedo_type);
                normal = n;
                break;
            }
            if (type == MAT_EMISSIVE) {
                const Color e = v3(__ldg(&params.materials[tri.w].emission_param));
                albedo = tint * Color(clampf(e.x, 0.f, 1.f), clampf(e.y, 0.f, 1.f), clampf(e.z, 0.f, 1.f));
                normal = n;
                break;
            }
            tint = tint * v3(albedo_type);
            Vec3 next_dir = reflect(dir, n);
            if (type == MAT_GLASS) {
                const float ior = __ldg(&params.materials[tri.w].emission_param).w;
                const float ratio = optixIsFrontFaceHit(optixHitObjectGetHitKind()) ? 1.f / ior : ior;
                const float cos_theta = fminf(dot(-dir, n), 1.f);
                const float sin2 = fmaxf(0.f, 1.f - cos_theta * cos_theta);
                if (ratio * ratio * sin2 <= 1.f) {
                    Vec3 perp = (dir + n * cos_theta) * ratio;
                    next_dir = perp + n * (-sqrtf(fmaxf(0.f, 1.f - length2(perp))));
                }
            }
            const float side = dot(next_dir, n) >= 0.f ? 1.f : -1.f;
            origin = hit_point + n * (side * kRayEps);
            dir = normalize(next_dir);
        }
        albedo_sum += albedo;
        normal_sum += normal;
    }

    const float inv = 1.f / (float)spp;
    const uint32_t pixel = pixel_index();
    params.albedo[pixel] = make_float4(albedo_sum.x * inv, albedo_sum.y * inv, albedo_sum.z * inv, 1.f);
    params.normal[pixel] = make_float4(normal_sum.x * inv, normal_sum.y * inv, normal_sum.z * inv, 0.f);
}
