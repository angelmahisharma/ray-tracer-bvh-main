#pragma once
#include <string>
#include <vector>
#include "color.h"

namespace rt {

// Float RGB image, linear colour. Row 0 is the TOP row. 3 floats per pixel.
struct Image {
    int width = 0, height = 0;
    std::vector<float> rgb;

    Image() = default;
    Image(int w, int h) : width(w), height(h), rgb((size_t)w * h * 3, 0.f) {}

    Color get(int x, int y) const {
        const float* p = &rgb[((size_t)y * width + x) * 3];
        return Color(p[0], p[1], p[2]);
    }
    void set(int x, int y, const Color& c) {
        float* p = &rgb[((size_t)y * width + x) * 3];
        p[0] = c.x; p[1] = c.y; p[2] = c.z;
    }
    float*       data()       { return rgb.data(); }
    const float* data() const { return rgb.data(); }
};

// 8-bit PNG. If apply_srgb is true the linear values are clamped to [0,1] and
// gamma-encoded (the default for "pretty" output). Pass false if the values
// are already display-ready. No external library needed.
bool save_png(const Image& img, const std::string& path, bool apply_srgb = true);

// Raw float image in PFM format (lossless, used for image comparison).
// Readable by Python (imageio / OpenCV), GIMP, Photoshop, Blender, etc.
bool save_pfm(const Image& img, const std::string& path);
bool load_pfm(const std::string& path, Image& out);

}  // namespace rt
