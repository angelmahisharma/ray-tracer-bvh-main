#include <glad/glad.h>   // must come before GLFW
#include <GLFW/glfw3.h>
#include <cuda_gl_interop.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include "cuda_check.h"
#include "renderer.h"
#include "rt/scene.h"

using namespace rt;

#ifdef _WIN32
// Hybrid-graphics laptops: run on the discrete GPU, so OpenGL and CUDA live on
// the same device (needed for CUDA/OpenGL buffer sharing).
extern "C" {
__declspec(dllexport) unsigned long NvOptimusEnablement = 1;
__declspec(dllexport) int AmdPowerXpressRequestHighPerformance = 1;
}
#endif

namespace {

// First-person camera: position + yaw/pitch, world up = +Y (team convention).
struct FlyCamera {
    Vec3 eye;
    float yaw = 0.f, pitch = 0.f;   // yaw 0 looks down -Z
    float fov = 45.f;

    Vec3 forward() const { return Vec3(sinf(yaw) * cosf(pitch), sinf(pitch), -cosf(yaw) * cosf(pitch)); }
    Vec3 right() const { return normalize(cross(forward(), Vec3(0.f, 1.f, 0.f))); }
    Vec3 up() const { return cross(right(), forward()); }

    void look_at(const Vec3& from, const Vec3& target) {
        eye = from;
        const Vec3 f = normalize(target - from);
        yaw = atan2f(f.x, -f.z);
        pitch = asinf(clampf(f.y, -1.f, 1.f));
    }
    Camera camera(float aspect) const {
        return make_camera(eye, eye + forward(), Vec3(0.f, 1.f, 0.f), fov, aspect);
    }
};

// Events collected by the GLFW callbacks, consumed once per frame.
struct Input {
    double scroll = 0.0;
    bool toggle_denoise = false, print_camera = false, reset = false;
    int spp_change = 0;   // -1 halve, +1 double
};

void key_callback(GLFWwindow* window, int key, int, int action, int) {
    if (action != GLFW_PRESS) return;
    Input& in = *static_cast<Input*>(glfwGetWindowUserPointer(window));
    switch (key) {
        case GLFW_KEY_ESCAPE: glfwSetWindowShouldClose(window, GLFW_TRUE); break;
        case GLFW_KEY_N: in.toggle_denoise = true; break;
        case GLFW_KEY_C: in.print_camera = true; break;
        case GLFW_KEY_R: in.reset = true; break;
        case GLFW_KEY_LEFT_BRACKET: in.spp_change = -1; break;
        case GLFW_KEY_RIGHT_BRACKET: in.spp_change = +1; break;
        default: break;
    }
}

void scroll_callback(GLFWwindow* window, double, double dy) {
    static_cast<Input*>(glfwGetWindowUserPointer(window))->scroll += dy;
}

// A w x h RGBA8 image on screen. CUDA writes straight into an OpenGL pixel
// buffer (zero copy); if CUDA/OpenGL sharing is unavailable it falls back to a
// device buffer copied through the host.
class Display {
public:
    Display(unsigned w, unsigned h) : w_((GLsizei)w), h_((GLsizei)h) {
        glGenTextures(1, &texture_);
        glBindTexture(GL_TEXTURE_2D, texture_);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w_, h_, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glGenFramebuffers(1, &fbo_);
        glBindFramebuffer(GL_READ_FRAMEBUFFER, fbo_);
        glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture_, 0);
        glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);

        glGenBuffers(1, &pbo_);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_);
        glBufferData(GL_PIXEL_UNPACK_BUFFER, (GLsizeiptr)w * h * sizeof(uchar4), nullptr, GL_STREAM_DRAW);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
        if (cudaGraphicsGLRegisterBuffer(&resource_, pbo_, cudaGraphicsMapFlagsWriteDiscard) != cudaSuccess) {
            cudaGetLastError();   // clear the error and use the host-copy path
            resource_ = nullptr;
            fallback_.alloc((size_t)w * h);
            host_.resize((size_t)w * h);
        }
    }
    ~Display() {
        if (resource_) cudaGraphicsUnregisterResource(resource_);
        glDeleteBuffers(1, &pbo_);
        glDeleteFramebuffers(1, &fbo_);
        glDeleteTextures(1, &texture_);
    }
    Display(const Display&) = delete;
    Display& operator=(const Display&) = delete;

    bool zero_copy() const { return resource_ != nullptr; }

    // Runs fill(uchar4* device_pixels) on `stream`, then updates the texture.
    template <typename Fill>
    void update(cudaStream_t stream, Fill fill) {
        glBindTexture(GL_TEXTURE_2D, texture_);
        if (resource_) {
            uchar4* pixels = nullptr;
            size_t bytes = 0;
            CUDA_CHECK(cudaGraphicsMapResources(1, &resource_, stream));
            CUDA_CHECK(cudaGraphicsResourceGetMappedPointer((void**)&pixels, &bytes, resource_));
            fill(pixels);
            CUDA_CHECK(cudaGraphicsUnmapResources(1, &resource_, stream));
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_);
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w_, h_, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
        } else {
            fill(fallback_.ptr);
            CUDA_CHECK(cudaMemcpyAsync(host_.data(), fallback_.ptr, host_.size() * sizeof(uchar4),
                                       cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w_, h_, GL_RGBA, GL_UNSIGNED_BYTE, host_.data());
        }
    }

    // Stretches the image over the window's framebuffer.
    void draw(int fb_width, int fb_height) {
        glBindFramebuffer(GL_READ_FRAMEBUFFER, fbo_);
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
        glBlitFramebuffer(0, 0, w_, h_, 0, 0, fb_width, fb_height, GL_COLOR_BUFFER_BIT, GL_LINEAR);
        glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
    }

private:
    GLsizei w_, h_;
    GLuint texture_ = 0, fbo_ = 0, pbo_ = 0;
    cudaGraphicsResource_t resource_ = nullptr;
    DeviceBuffer<uchar4> fallback_;
    std::vector<uchar4> host_;
};

void usage() {
    std::cerr << "Usage: optix_viewer <scene.scene> [options]\n"
                 "  --spp N          samples per pixel per frame (default 4)\n"
                 "  --max-bounces N  override the scene's max_bounces\n"
                 "  --no-denoise     start with the denoiser off (toggle with N)\n";
}

}  // namespace

int main(int argc, char** argv) {
    std::string scene_path;
    int spp = 4, max_bounces = -1;
    bool denoise_on = true;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--spp" && i + 1 < argc) spp = std::max(1, std::atoi(argv[++i]));
        else if (a == "--max-bounces" && i + 1 < argc) max_bounces = std::max(0, std::atoi(argv[++i]));
        else if (a == "--no-denoise") denoise_on = false;
        else if (a.rfind("--", 0) == 0 || !scene_path.empty()) { usage(); return 1; }
        else scene_path = a;
    }
    if (scene_path.empty()) { usage(); return 1; }

    Scene scene;
    std::string error;
    if (!load_scene(scene_path, scene, &error)) {
        std::cerr << "Scene load failed: " << error << '\n';
        return 1;
    }
    scene.settings.spp = spp;
    if (max_bounces >= 0) scene.settings.max_bounces = max_bounces;
    const unsigned w = (unsigned)scene.settings.width, h = (unsigned)scene.settings.height;

    if (!glfwInit()) {
        std::cerr << "glfwInit failed\n";
        return 1;
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    GLFWwindow* window = glfwCreateWindow((int)w, (int)h, "optix_viewer", nullptr, nullptr);
    if (!window) {
        std::cerr << "Could not create an OpenGL 3.3 window\n";
        glfwTerminate();
        return 1;
    }
    glfwMakeContextCurrent(window);
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "Could not load OpenGL functions\n";
        glfwTerminate();
        return 1;
    }
    glfwSwapInterval(0);   // no vsync: the title shows the real frame time
    Input input;
    glfwSetWindowUserPointer(window, &input);
    glfwSetKeyCallback(window, key_callback);
    glfwSetScrollCallback(window, scroll_callback);

    int exit_code = 0;
    try {
        RenderOptions options;
        options.denoise = true;   // always set up, so N can toggle it
        options.guide_spp = 1;    // guides are recomputed after every camera move: keep them cheap
        Renderer renderer(scene, options);
        Display display(renderer.out_width(), renderer.out_height());

        std::printf("%s: %zu triangles, %ux%u, %d spp/frame, max %d bounces, display %s\n", scene_path.c_str(),
                    scene.triangles.size(), w, h, spp, scene.settings.max_bounces,
                    display.zero_copy() ? "zero-copy (CUDA/OpenGL interop)" : "via host copy");
        std::printf("left drag: look | right/middle drag: pan | scroll: dolly | WASD + Q/E: move | Shift: fast\n"
                    "[ / ]: spp per frame | N: denoiser | C: print camera | R: reset | Esc: quit\n");
        std::fflush(stdout);

        FlyCamera start;
        start.look_at(scene.cam_eye, scene.cam_target);
        start.fov = scene.cam_fov;
        FlyCamera cam = start;
        const float aspect = (float)w / (float)h;
        const float base_speed = std::max(length(scene.bounds.extent()) * 0.15f, 1e-3f);   // units per second

        uint32_t frame = 0;
        bool guides_valid = false;
        double last_time = glfwGetTime(), title_time = last_time;
        int title_frames = 0;
        double last_x = 0.0, last_y = 0.0;
        glfwGetCursorPos(window, &last_x, &last_y);

        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();
            const double now = glfwGetTime();
            const float dt = (float)(now - last_time);
            last_time = now;

            // ---- camera controls ----
            bool moved = false;
            double mx = 0.0, my = 0.0;
            glfwGetCursorPos(window, &mx, &my);
            const float dx = (float)(mx - last_x), dy = (float)(my - last_y);
            last_x = mx;
            last_y = my;
            const bool fast = glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS ||
                              glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS;
            const float speed = base_speed * (fast ? 5.f : 1.f);
            if (dx != 0.f || dy != 0.f) {
                if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS) {
                    cam.yaw += dx * 0.004f;
                    cam.pitch = clampf(cam.pitch - dy * 0.004f, -1.55f, 1.55f);
                    moved = true;
                } else if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS ||
                           glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS) {
                    cam.eye += (cam.right() * -dx + cam.up() * dy) * (speed * 0.002f);
                    moved = true;
                }
            }
            if (input.scroll != 0.0) {
                cam.eye += cam.forward() * ((float)input.scroll * speed * 0.1f);
                input.scroll = 0.0;
                moved = true;
            }
            Vec3 move(0.f);
            if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) move += cam.forward();
            if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) move -= cam.forward();
            if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) move += cam.right();
            if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) move -= cam.right();
            if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS) move += Vec3(0.f, 1.f, 0.f);
            if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS) move -= Vec3(0.f, 1.f, 0.f);
            if (length2(move) > 0.f) {
                cam.eye += normalize(move) * (speed * dt);
                moved = true;
            }
            if (input.reset) {
                cam = start;
                input.reset = false;
                moved = true;
            }
            if (input.spp_change != 0) {
                spp = std::clamp(input.spp_change > 0 ? spp * 2 : spp / 2, 1, 256);
                renderer.set_spp((uint32_t)spp);
                input.spp_change = 0;
                moved = true;   // restart accumulation so every frame in the mean has the same weight
            }
            if (input.toggle_denoise) {
                denoise_on = !denoise_on;
                input.toggle_denoise = false;
            }
            if (input.print_camera) {
                const Vec3 t = cam.eye + cam.forward();
                std::printf("camera eye %.4f %.4f %.4f target %.4f %.4f %.4f up 0 1 0 fov %.2f\n", cam.eye.x,
                            cam.eye.y, cam.eye.z, t.x, t.y, t.z, cam.fov);
                std::fflush(stdout);
                input.print_camera = false;
            }
            if (moved) {
                renderer.set_camera(cam.camera(aspect));
                frame = 0;
                guides_valid = false;
            }

            // ---- render: path trace (averaged into previous frames), denoise, show ----
            renderer.set_frame(frame, 1234u + frame * 0x9E3779B9u);
            renderer.trace();
            if (denoise_on) {
                if (!guides_valid) {
                    renderer.trace_guides();
                    guides_valid = true;
                }
                renderer.denoise();
            }
            const float4* shown = denoise_on ? renderer.denoised() : renderer.accum();
            display.update(renderer.stream(), [&](uchar4* pixels) { renderer.to_display(shown, pixels); });
            int fb_width = 0, fb_height = 0;
            glfwGetFramebufferSize(window, &fb_width, &fb_height);
            display.draw(fb_width, fb_height);
            glfwSwapBuffers(window);
            ++frame;

            // ---- window title, averaged over half a second ----
            ++title_frames;
            if (now - title_time >= 0.5) {
                const double ms = (now - title_time) * 1000.0 / title_frames;
                char title[256];
                std::snprintf(title, sizeof(title),
                              "optix_viewer | %d spp/frame | %u frames accumulated | %.1f ms (%.0f fps) | denoiser %s",
                              spp, frame, ms, 1000.0 / ms, denoise_on ? "on" : "off");
                glfwSetWindowTitle(window, title);
                title_time = now;
                title_frames = 0;
            }
        }
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        exit_code = 1;
    }
    glfwDestroyWindow(window);
    glfwTerminate();
    return exit_code;
}
