#pragma once
#include "vec3.h"

namespace rt {

// Axis-aligned bounding box. Starts "empty" (lo > hi) so expand() just works.
struct AABB {
    Vec3 lo{kBigFloat, kBigFloat, kBigFloat};
    Vec3 hi{-kBigFloat, -kBigFloat, -kBigFloat};

    RT_HD void expand(const Vec3& p)  { lo = vmin(lo, p);    hi = vmax(hi, p); }
    RT_HD void expand(const AABB& b)  { lo = vmin(lo, b.lo); hi = vmax(hi, b.hi); }
    RT_HD bool valid() const { return lo.x <= hi.x && lo.y <= hi.y && lo.z <= hi.z; }
    RT_HD Vec3 extent() const { return hi - lo; }
    RT_HD Vec3 center() const { return (lo + hi) * 0.5f; }
    RT_HD float surface_area() const {
        Vec3 e = extent();
        return 2.f * (e.x * e.y + e.y * e.z + e.z * e.x);
    }
    RT_HD int longest_axis() const {
        Vec3 e = extent();
        return (e.x > e.y && e.x > e.z) ? 0 : (e.y > e.z ? 1 : 2);
    }
};

}  // namespace rt
