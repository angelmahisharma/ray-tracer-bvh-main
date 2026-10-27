#pragma once
#include "vec3.h"

namespace rt {

using Color = Vec3;   // linear RGB, values may exceed 1 (HDR)

RT_HD inline float luminance(const Color& c) { return 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z; }

// sRGB transfer function (linear -> display). Part 7 may replace the
// tone mapping used when saving PNGs; this is the default.
RT_HD inline float linear_to_srgb(float c) {
    c = clampf(c, 0.f, 1.f);
    return c <= 0.0031308f ? 12.92f * c : 1.055f * powf(c, 1.f / 2.4f) - 0.055f;
}

}  // namespace rt
