// CUDA path tracer with a CPU-built, GPU-traversed BVH (Part 6 bonus).
// Same CLI contract as cpu_renderer / optix_renderer:
//   cuda_bvh_renderer <scene.scene> [output_prefix] [spp] [max_bounces] [options]
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "render.cuh"
#include "rt/image.h"

using namespace rt;

#define CUDA_CHECK(call)                                                                      \
    do {                                                                                      \
        cudaError_t e_ = (call);                                                              \
        if (e_ != cudaSuccess) {                                                              \
            std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n", cudaGetErrorName(e_), __FILE__, \
                         __LINE__, cudaGetErrorString(e_));                                   \
            std::exit(1);                                                                     \
        }                                                                                     \
    } while (0)

static void usage() {
    std::cerr << "Usage: cuda_bvh_renderer <scene.scene> [output_prefix] [spp] [max_bounces] [options]\n"
                 "  --runs N       timed renders after one warm-up (default 3, median reported)\n"
                 "  --brute        no BVH: test every triangle (the Part 4 baseline)\n"
                 "  --leaf N       max triangles per BVH leaf (default 4)\n"
                 "  --rows N       image rows per kernel launch (default 45; keeps launches short\n"
                 "                 so the display watchdog does not kill them)\n"
                 "  --seed S       global RNG seed (default 1234, same as cpu_renderer)\n";
}

template <class T>
static T* upload(const std::vector<T>& v) {
    T* d = nullptr;
    size_t bytes = std::max<size_t>(v.size(), 1) * sizeof(T);
    CUDA_CHECK(cudaMalloc(&d, bytes));
    if (!v.empty()) CUDA_CHECK(cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
    return d;
}

int main(int argc, char** argv) {
    std::vector<std::string> positional;
    int runs = 3, leaf = 4, rows_per_launch = 45;
    bool brute = false;
    uint64_t seed = 1234;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--runs" && i + 1 < argc) runs = std::max(1, std::atoi(argv[++i]));
        else if (a == "--leaf" && i + 1 < argc) leaf = std::max(1, std::atoi(argv[++i]));
        else if (a == "--rows" && i + 1 < argc) rows_per_launch = std::max(1, std::atoi(argv[++i]));
        else if (a == "--seed" && i + 1 < argc) seed = std::strtoull(argv[++i], nullptr, 10);
        else if (a == "--brute") brute = true;
        else if (a.rfind("--", 0) == 0) { usage(); return 1; }
        else positional.push_back(a);
    }
    if (positional.empty()) { usage(); return 1; }

    Scene scene;
    std::string error;
    if (!load_scene(positional[0], scene, &error)) {
        std::cerr << "Scene load failed: " << error << '\n';
        return 1;
    }
    if (positional.size() >= 3) scene.settings.spp = std::max(1, std::atoi(positional[2].c_str()));
    if (positional.size() >= 4) scene.settings.max_bounces = std::max(0, std::atoi(positional[3].c_str()));
    const std::string prefix = positional.size() >= 2 ? positional[1] : "output";

    cudaDeviceProp prop = {};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    // ---- BVH build on the CPU (reported separately, never part of the render time) ----
    PreparedScene ps;
    auto b0 = std::chrono::steady_clock::now();
    prepare_scene(scene, ps, leaf);
    double build_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - b0).count();

    // ---- upload ----
    auto u0 = std::chrono::steady_clock::now();
    SceneView dev = ps.host_view(!brute);
    dev.seed = seed;
    BVHNode* d_nodes = upload(ps.bvh.nodes);
    Triangle* d_tris = upload(brute ? ps.plain_tris : ps.bvh.tris);
    Material* d_mats = upload(ps.materials);
    Light* d_lights = upload(brute ? ps.plain_lights : ps.lights);
    int32_t* d_tri_light = upload(brute ? ps.plain_tri_light : ps.tri_light);
    dev.nodes = d_nodes; dev.tris = d_tris; dev.materials = d_mats; dev.lights = d_lights; dev.tri_light = d_tri_light;
    const int w = dev.width, h = dev.height;
    float4* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_out, (size_t)w * h * sizeof(float4)));
    CUDA_CHECK(cudaMemset(d_out, 0, (size_t)w * h * sizeof(float4)));
    CUDA_CHECK(cudaDeviceSynchronize());
    double upload_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - u0).count();

    // ---- timed renders: one warm-up, then `runs` full frames (GPU events, launches in row bands) ----
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    cudaEvent_t ev0, ev1;
    CUDA_CHECK(cudaEventCreate(&ev0));
    CUDA_CHECK(cudaEventCreate(&ev1));
    std::vector<float> times;
    for (int r = -1; r < runs; ++r) {
        CUDA_CHECK(cudaEventRecord(ev0, stream));
        for (int y0 = 0; y0 < h; y0 += rows_per_launch) {
            launch_render(dev, d_out, y0, std::min(rows_per_launch, h - y0), stream);
            CUDA_CHECK(cudaGetLastError());
        }
        CUDA_CHECK(cudaEventRecord(ev1, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        if (r < 0) continue;
        float ms = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
        times.push_back(ms);
    }
    std::vector<float> sorted = times;
    std::sort(sorted.begin(), sorted.end());
    const float median_ms = sorted[sorted.size() / 2];

    // ---- output ----
    std::vector<float4> px((size_t)w * h);
    CUDA_CHECK(cudaMemcpy(px.data(), d_out, px.size() * sizeof(float4), cudaMemcpyDeviceToHost));
    Image img(w, h);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            const float4& p = px[(size_t)y * w + x];
            img.set(x, y, Color(p.x, p.y, p.z));
        }
    const bool ok = save_pfm(img, prefix + ".pfm") && save_png(img, prefix + ".png");

    const double samples = (double)w * h * scene.settings.spp;
    std::printf("GPU: %s\n", prop.name);
    std::printf("Triangles: %zu\nResolution: %dx%d\nSPP: %d\nMax bounces: %d\n", scene.triangles.size(), w, h,
                scene.settings.spp, scene.settings.max_bounces);
    if (brute) {
        std::printf("Acceleration: none (brute force)\n");
    } else {
        std::printf("BVH build: %.3f ms (CPU, binned SAH, %zu nodes, %d leaves, depth %d, SAH cost %.1f, leaf<=%d)\n",
                    build_ms, ps.bvh.nodes.size(), ps.bvh.leaf_count, ps.bvh.max_depth, ps.bvh.sah_cost, leaf);
        std::printf("BVH memory: %.1f KiB\n", ps.bvh.nodes.size() * sizeof(BVHNode) / 1024.0);
    }
    std::printf("Upload: %.3f ms\n", upload_ms);
    std::printf("Render time: %.6f s (median of %d, min %.3f ms, max %.3f ms)\n", median_ms / 1000.0, runs,
                sorted.front(), sorted.back());
    std::printf("Samples/s: %.1f M\n", samples / (median_ms * 1e-3) / 1e6);

    cudaFree(d_nodes); cudaFree(d_tris); cudaFree(d_mats); cudaFree(d_lights); cudaFree(d_tri_light); cudaFree(d_out);
    cudaEventDestroy(ev0); cudaEventDestroy(ev1); cudaStreamDestroy(stream);
    if (!ok) { std::cerr << "Failed to save output image(s).\n"; return 1; }
    std::printf("Saved %s.png and %s.pfm\n", prefix.c_str(), prefix.c_str());
    return 0;
}
