#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include "cuda_check.h"
#include "renderer.h"
#include "rt/image.h"
#include "rt/scene.h"

using namespace rt;

static void usage() {
    std::cerr << "Usage: optix_renderer <scene.scene> [output_prefix] [spp] [max_bounces] [options]\n"
                 "  --runs N        timed frames after one warm-up (default 5, median reported)\n"
                 "  --count-rays    extra launch that counts traced rays (for Grays/s)\n"
                 "  --no-ser        disable Shader Execution Reordering\n"
                 "  --maxreg N      cap registers per thread (0 = OptiX default)\n"
                 "  --seed S        global RNG seed (default 1234, same as cpu_renderer)\n"
                 "  --denoise       run the OptiX AI denoiser, also write <prefix>_denoised.png/.pfm\n"
                 "  --guide-spp N   jittered samples for the denoiser's albedo/normal guides (default 4)\n"
                 "  --save-guides   also write <prefix>_albedo.png and <prefix>_normal.png\n"
                 "  --upscale2x     path-trace at half width/height, denoiser upscales to full size\n"
                 "  --verbose       OptiX log level 4\n";
}

static float median(std::vector<float> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

// Downloads a width*height float4 buffer into an Image (RGB, row 0 = top).
static Image download(const float4* d_buffer, unsigned w, unsigned h, float scale = 1.f, float offset = 0.f) {
    std::vector<float4> pixels((size_t)w * h);
    CUDA_CHECK(cudaMemcpy(pixels.data(), d_buffer, pixels.size() * sizeof(float4), cudaMemcpyDeviceToHost));
    Image img((int)w, (int)h);
    for (unsigned y = 0; y < h; ++y)
        for (unsigned x = 0; x < w; ++x) {
            const float4& p = pixels[(size_t)y * w + x];
            img.set((int)x, (int)y, Color(p.x * scale + offset, p.y * scale + offset, p.z * scale + offset));
        }
    return img;
}

int main(int argc, char** argv) {
    std::vector<std::string> positional;
    RenderOptions options;
    int runs = 5;
    bool count_rays = false, save_guides = false;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--runs" && i + 1 < argc) runs = std::max(1, std::atoi(argv[++i]));
        else if (a == "--maxreg" && i + 1 < argc) options.max_registers = (unsigned)std::max(0, std::atoi(argv[++i]));
        else if (a == "--seed" && i + 1 < argc) options.seed = (uint32_t)std::strtoul(argv[++i], nullptr, 10);
        else if (a == "--guide-spp" && i + 1 < argc) options.guide_spp = (uint32_t)std::max(1, std::atoi(argv[++i]));
        else if (a == "--count-rays") count_rays = true;
        else if (a == "--no-ser") options.reorder = false;
        else if (a == "--denoise") options.denoise = true;
        else if (a == "--save-guides") save_guides = options.denoise = true;
        else if (a == "--upscale2x") options.upscale2x = options.denoise = true;
        else if (a == "--verbose") options.verbose = true;
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

    // ---- setup (not timed; the GAS build is timed separately inside) ----
    try {
        Renderer renderer(scene, options);
        const bool denoise = renderer.denoising();
        cudaStream_t stream = renderer.stream();

        // ---- frames (timed per stage on the GPU, after one warm-up) ----
        // A frame = path tracing [+ guide pass + denoiser]; this is the real-time budget.
        cudaEvent_t ev[4];
        for (cudaEvent_t& e : ev) CUDA_CHECK(cudaEventCreate(&e));
        std::vector<float> render_ms, guides_ms, denoise_ms, frame_ms;
        for (int r = -1; r < runs; ++r) {
            CUDA_CHECK(cudaEventRecord(ev[0], stream));
            renderer.trace();
            CUDA_CHECK(cudaEventRecord(ev[1], stream));
            if (denoise) {
                renderer.trace_guides();
                CUDA_CHECK(cudaEventRecord(ev[2], stream));
                renderer.denoise();
                CUDA_CHECK(cudaEventRecord(ev[3], stream));
            }
            CUDA_CHECK(cudaStreamSynchronize(stream));
            if (r < 0) continue;   // warm-up
            float ms = 0.f;
            CUDA_CHECK(cudaEventElapsedTime(&ms, ev[0], ev[1]));
            render_ms.push_back(ms);
            if (denoise) {
                CUDA_CHECK(cudaEventElapsedTime(&ms, ev[1], ev[2]));
                guides_ms.push_back(ms);
                CUDA_CHECK(cudaEventElapsedTime(&ms, ev[2], ev[3]));
                denoise_ms.push_back(ms);
                CUDA_CHECK(cudaEventElapsedTime(&ms, ev[0], ev[3]));
                frame_ms.push_back(ms);
            }
        }
        for (cudaEvent_t& e : ev) CUDA_CHECK(cudaEventDestroy(e));
        const float render_median = median(render_ms);
        const auto minmax = std::minmax_element(render_ms.begin(), render_ms.end());
        // Separate instrumented launch so counting never affects the timed runs.
        const unsigned long long ray_count = count_rays ? renderer.count_rays() : 0;

        // ---- output (the raw .pfm is always the un-denoised radiance, for comparisons) ----
        const unsigned w = renderer.width(), h = renderer.height();
        const Image img = download(renderer.accum(), w, h);
        bool ok = save_pfm(img, prefix + ".pfm") && save_png(img, prefix + ".png");
        if (denoise) {
            const Image den = download(renderer.denoised(), renderer.out_width(), renderer.out_height());
            ok = ok && save_pfm(den, prefix + "_denoised.pfm") && save_png(den, prefix + "_denoised.png");
        }
        if (save_guides) {
            ok = ok && save_png(download(renderer.albedo(), w, h), prefix + "_albedo.png") &&
                 save_png(download(renderer.normal(), w, h, 0.5f, 0.5f), prefix + "_normal.png", false);
        }

        const Gas& gas = renderer.gas();
        const double samples = (double)w * h * scene.settings.spp;
        std::printf("GPU: %s\n", prop.name);
        std::printf("Triangles: %zu\nResolution: %ux%u", scene.triangles.size(), w, h);
        if (options.upscale2x) std::printf(" (denoiser upscales to %ux%u)", renderer.out_width(), renderer.out_height());
        std::printf("\nSPP: %d\nMax bounces: %d\n", scene.settings.spp, scene.settings.max_bounces);
        std::printf("GAS build: %.3f ms (%.1f KiB compacted, %.1f KiB before)\n", gas.build_ms, gas.bytes / 1024.0,
                    gas.uncompacted_bytes / 1024.0);
        std::printf("SER: %s, maxreg: %u\n", options.reorder ? "on" : "off", options.max_registers);
        std::printf("Render time: %.6f s (median of %d, min %.3f ms, max %.3f ms)\n", render_median / 1000.0, runs,
                    *minmax.first, *minmax.second);
        std::printf("Samples/s: %.1f M\n", samples / (render_median * 1e-3) / 1e6);
        if (count_rays)
            std::printf("Rays: %.1f M (%.2f per sample), %.2f Grays/s\n", ray_count / 1e6, ray_count / samples,
                        ray_count / (render_median * 1e-3) / 1e9);
        if (denoise) {
            const float frame = median(frame_ms);
            std::printf("Guide pass: %.3f ms (%u spp)\nDenoise: %.3f ms\n", median(guides_ms), options.guide_spp,
                        median(denoise_ms));
            std::printf("Frame time (render + guides + denoise): %.3f ms = %.1f fps\n", frame, 1000.f / frame);
        }
        if (!ok) {
            std::cerr << "Failed to save output image(s).\n";
            return 1;
        }
        std::printf("Saved %s.png and %s.pfm%s\n", prefix.c_str(), prefix.c_str(),
                    denoise ? (" (+ " + prefix + "_denoised)").c_str() : "");
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
    return 0;
}
