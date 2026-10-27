#pragma once
#include "vec3.h"

namespace rt {

// Valid hits satisfy tmin < t < tmax. 32 bytes, GPU friendly.
struct Ray {
    Vec3  origin;
    float tmin = kRayEps;
    Vec3  dir;            // should be normalized
    float tmax = kBigFloat;
    Ray() = default;
    RT_HD Ray(const Vec3& o, const Vec3& d, float tmin_ = kRayEps, float tmax_ = kBigFloat)
        : origin(o), tmin(tmin_), dir(d), tmax(tmax_) {}
    RT_HD Vec3 at(float t) const { return origin + dir * t; }
};

}  // namespace rt
