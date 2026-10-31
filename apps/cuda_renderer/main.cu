// Renderer B: CUDA path tracer, NO BVH. Every ray checks every triangle.
// A direct port of apps/cpu_renderer/main.cpp so both produce the same image.
#include "rt/scene.h"
#include "rt/image.h"
#include "rt/rng.h"

#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

using namespace rt;

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t e_ = (call);                                               \
        if (e_ != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error: %s (%s:%d)\n",                        \
                    cudaGetErrorString(e_), __FILE__, __LINE__);               \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

// Everything the kernel needs, as GPU pointers.
struct DevScene {
    const Triangle* tris;   int num_tris;
    const Material* mats;   int num_mats;
    const Light*    lights; int num_lights;
    float           total_light_power;
    int             max_bounces;
    Environment     env;
    Camera          cam;
};

struct Hit {
    float t = kBigFloat;
    float u = 0.f, v = 0.f;
    uint32_t tri_index = 0;
    Vec3 p, normal;
    bool front_face = true;
};

// ---------- brute force: loops over EVERY triangle (same maths as the CPU) ----------
__device__ bool closest_hit(const DevScene& sc, const Ray& ray, Hit& hit) {
    bool found = false;
    float closest = ray.tmax;
    for (uint32_t i = 0; i < (uint32_t)sc.num_tris; ++i) {
        const Triangle& tri = sc.tris[i];
        Vec3 e1 = tri.v1 - tri.v0, e2 = tri.v2 - tri.v0;
        Vec3 pvec = cross(ray.dir, e2);
        float det = dot(e1, pvec);
        if (fabsf(det) < 1e-8f) continue;
        float inv_det = 1.f / det;
        Vec3 tvec = ray.origin - tri.v0;
        float u = dot(tvec, pvec) * inv_det;
        if (u < 0.f || u > 1.f) continue;
        Vec3 qvec = cross(tvec, e1);
        float v = dot(ray.dir, qvec) * inv_det;
        if (v < 0.f || u + v > 1.f) continue;
        float t = dot(e2, qvec) * inv_det;
        if (t <= ray.tmin || t >= closest) continue;
        closest = t;
        found = true;
        hit.t = t; hit.u = u; hit.v = v; hit.tri_index = i;
    }
    if (found) {   // work out position and normals once, for the winning triangle
        const Triangle& tri = sc.tris[hit.tri_index];
        hit.p = ray.at(hit.t);
        hit.front_face = dot(ray.dir, tri.geometric_normal()) < 0.f;
        Vec3 n = tri.shading_normal(hit.u, hit.v);
        if (dot(n, ray.dir) > 0.f) n = -n;
        hit.normal = n;
    }
    return found;
}

__device__ float power_heuristic(float a, float b) {
    float aa = a * a, bb = b * b;
    return aa / fmaxf(aa + bb, 1e-20f);
}

__device__ Vec3 random_in_unit_sphere(Rng& rng) {
    for (int i = 0; i < 100; ++i) {
        float x = 2.f * rng.next_float() - 1.f;
        float y = 2.f * rng.next_float() - 1.f;
        float z = 2.f * rng.next_float() - 1.f;
        Vec3 p(x, y, z);
        if (length2(p) < 1.f) return p;
    }
    return Vec3(0.f, 0.f, 0.f);
}

__device__ Vec3 random_unit_vector(Rng& rng) { return normalize(random_in_unit_sphere(rng)); }

__device__ float schlick(float cosine, float ref_idx) {
    float r0 = (1.f - ref_idx) / (1.f + ref_idx);
    r0 *= r0;
    return r0 + (1.f - r0) * powf(1.f - clampf(cosine, 0.f, 1.f), 5.f);
}

__device__ bool sample_light(const DevScene& sc, const Hit& surface, Rng& rng, Color& contribution) {
    contribution = Color(0.f);
    if (sc.num_lights == 0 || sc.total_light_power <= 0.f) return false;

    float pick = rng.next_float();
    const Light* light = &sc.lights[sc.num_lights - 1];
    for (int i = 0; i < sc.num_lights; ++i) {
        if (pick <= sc.lights[i].cdf) { light = &sc.lights[i]; break; }
    }
    if (light->tri_index >= (uint32_t)sc.num_tris || light->area <= 0.f || light->power <= 0.f) return false;

    const Triangle& light_tri = sc.tris[light->tri_index];
    const Material& light_mat = sc.mats[light_tri.material_id];
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
    if (shadow.tmax <= shadow.tmin || closest_hit(sc, shadow, blocker)) return false;

    float select_pdf = light->power / sc.total_light_power;
    float light_pdf = select_pdf * dist2 / (light->area * cos_light);
    if (light_pdf <= 0.f) return false;
    float bsdf_pdf = cos_surface / kPi;
    float mis = power_heuristic(light_pdf, bsdf_pdf);
    Color bsdf = light_mat.emission * (cos_surface / (kPi * light_pdf));
    contribution = bsdf * mis;   // the caller multiplies by the diffuse albedo
    return true;
}

__device__ float light_pdf_for_hit(const DevScene& sc, uint32_t tri_index, const Vec3& from,
                                   const Vec3& hit_point, const Vec3& ray_dir) {
    for (int i = 0; i < sc.num_lights; ++i) {
        const Light& light = sc.lights[i];
        if (light.tri_index != tri_index || light.area <= 0.f || sc.total_light_power <= 0.f) continue;
        const Triangle& tri = sc.tris[tri_index];
        float cos_light = dot(tri.geometric_normal(), -ray_dir);
        if (cos_light <= 0.f) return 0.f;
        float dist2 = length2(hit_point - from);
        float select_pdf = light.power / sc.total_light_power;
        return select_pdf * dist2 / (light.area * cos_light);
    }
    return 0.f;
}

// ---------- the path tracing loop: a loop (no recursion), same rules as the CPU ----------
__device__ Color trace(const DevScene& sc, Ray ray, Rng& rng) {
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
        if (tri.material_id >= (uint32_t)sc.num_mats) break;
        const Material& mat = sc.mats[tri.material_id];

        if (mat.type == MAT_EMISSIVE) {
            float weight = 1.f;
            if (previous_was_diffuse) {
                float light_pdf = light_pdf_for_hit(sc, hit.tri_index, previous_point, hit.p, normalize(ray.dir));
                weight = power_heuristic(previous_bsdf_pdf, light_pdf);
            }
            if (hit.front_face || bounce == 0)
                radiance += throughput * mat.emission * weight;
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

// ---------- the kernel: one thread = one pixel ----------
__global__ void renderKernel(DevScene sc, Color* accum, int W, int H,
                             int sample_start, int sample_count, unsigned seed) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;                    // spare threads at the edge do nothing

    uint64_t idx  = (uint64_t)y * (uint64_t)W + (uint64_t)x;
    uint64_t npix = (uint64_t)W * (uint64_t)H;
    Color sum(0.f);

    for (int k = 0; k < sample_count; ++k) {
        Rng rng;
        rng.seed(seed, (uint64_t)(sample_start + k) * npix + idx);   // unique stream per pixel and sample
        float jx = rng.next_float();
        float jy = rng.next_float();
        Ray ray = pixel_ray(sc.cam, x, y, W, H, jx, jy);             // pixel (0,0) is top-left
        Color c = trace(sc, ray, rng);
        if (c.x == c.x && c.y == c.y && c.z == c.z) sum += c;        // drop NaN samples
    }
    accum[idx] += sum;
}

// ======================= host side =======================
template <class T>
static T* upload(const std::vector<T>& v) {
    T* d = nullptr;
    CUDA_CHECK(cudaMalloc(&d, std::max<size_t>(v.size() * sizeof(T), 1)));
    if (!v.empty())
        CUDA_CHECK(cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
    return d;
}

// One full render. Returns GPU time in ms (render only).
static float render_once(const DevScene& sc, Color* d_accum, int W, int H, int spp, int chunk, unsigned seed) {
    CUDA_CHECK(cudaMemset(d_accum, 0, sizeof(Color) * (size_t)W * H));
    dim3 block(16, 16);
    dim3 grid((W + 15) / 16, (H + 15) / 16);

    cudaEvent_t a, b;
    CUDA_CHECK(cudaEventCreate(&a));
    CUDA_CHECK(cudaEventCreate(&b));
    CUDA_CHECK(cudaEventRecord(a));
    for (int s = 0; s < spp; s += chunk) {            // small launches keep the OS watchdog happy
        int n = std::min(chunk, spp - s);
        renderKernel<<<grid, block>>>(sc, d_accum, W, H, s, n, seed);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaEventRecord(b));
    CUDA_CHECK(cudaEventSynchronize(b));              // wait until the GPU is really done
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
    CUDA_CHECK(cudaEventDestroy(a));
    CUDA_CHECK(cudaEventDestroy(b));
    return ms;
}

int main(int argc, char** argv) {
    // Usage: cuda_renderer <scene.scene> [output_prefix] [spp] [max_bounces] [--runs N] [--chunk N]
    std::vector<std::string> pos;
    int runs = 5, chunk = 8;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--runs" && i + 1 < argc)       runs  = std::max(1, std::atoi(argv[++i]));
        else if (a == "--chunk" && i + 1 < argc) chunk = std::max(1, std::atoi(argv[++i]));
        else pos.push_back(a);
    }
    if (pos.empty()) {
        std::cerr << "Usage: cuda_renderer <scene.scene> [output_prefix] [spp] [max_bounces] [--runs N] [--chunk N]\n";
        return 1;
    }

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s | %.0f MB\n", prop.name, prop.totalGlobalMem / 1048576.0);

    // ---- load with Part 1's loader (not timed) ----
    Scene scene; std::string error;
    if (!load_scene(pos[0], scene, &error)) { std::cerr << "Scene load failed: " << error << '\n'; return 1; }
    if (pos.size() >= 3) scene.settings.spp = std::max(1, std::atoi(pos[2].c_str()));
    if (pos.size() >= 4) scene.settings.max_bounces = std::max(0, std::atoi(pos[3].c_str()));
    std::string prefix = pos.size() >= 2 ? pos[1] : "output";
    const int W = scene.settings.width, H = scene.settings.height, spp = scene.settings.spp;

    // ---- upload to the GPU (timed separately, NOT part of render time) ----
    auto u0 = std::chrono::steady_clock::now();
    DevScene sc;
    sc.tris   = upload(scene.triangles);  sc.num_tris   = (int)scene.triangles.size();
    sc.mats   = upload(scene.materials);  sc.num_mats   = (int)scene.materials.size();
    sc.lights = upload(scene.lights);     sc.num_lights = (int)scene.lights.size();
    sc.total_light_power = scene.total_light_power;
    sc.max_bounces = scene.settings.max_bounces;
    sc.env = scene.env;
    sc.cam = scene.camera;
    Color* d_accum = nullptr;
    CUDA_CHECK(cudaMalloc(&d_accum, sizeof(Color) * (size_t)W * H));
    CUDA_CHECK(cudaDeviceSynchronize());
    double upload_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - u0).count();

    // ---- warm-up run (thrown away), then timed runs ----
    const unsigned seed = 1234;
    render_once(sc, d_accum, W, H, 1, chunk, seed);
    std::vector<float> times;
    for (int r = 0; r < runs; ++r) {
        times.push_back(render_once(sc, d_accum, W, H, spp, chunk, seed));
        printf("run %d: %.2f ms\n", r + 1, times.back());
    }
    double mean_ms = 0; for (float t : times) mean_ms += t; mean_ms /= times.size();

    // ---- copy back (after timing), average, save ----
    std::vector<Color> host((size_t)W * H);
    CUDA_CHECK(cudaMemcpy(host.data(), d_accum, sizeof(Color) * host.size(), cudaMemcpyDeviceToHost));
    Image img(W, H);
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x)
            img.set(x, y, host[(size_t)y * W + x] / (float)spp);
    const bool pfm_ok = save_pfm(img, prefix + ".pfm");
    const bool png_ok = save_png(img, prefix + ".png");

    // Same style of lines as the CPU renderer, so Part 8's runner can read them.
    std::cout << "Triangles: " << scene.triangles.size() << "\nResolution: " << W << "x" << H
              << "\nSPP: " << spp << "\nMax bounces: " << scene.settings.max_bounces
              << "\nRender time: " << mean_ms / 1000.0 << " s\n"
              << "Upload time: " << upload_ms / 1000.0 << " s\n"
              << "Render time min/max: " << *std::min_element(times.begin(), times.end()) / 1000.0
              << " / " << *std::max_element(times.begin(), times.end()) / 1000.0 << " s\n";
    if (!pfm_ok || !png_ok) { std::cerr << "Failed to save output image(s).\n"; return 1; }
    std::cout << "Saved " << prefix << ".png and " << prefix << ".pfm\n";

    cudaFree((void*)sc.tris); cudaFree((void*)sc.mats); cudaFree((void*)sc.lights); cudaFree(d_accum);
    return 0;
}