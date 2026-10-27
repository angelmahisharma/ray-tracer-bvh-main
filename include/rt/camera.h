#pragma once
#include "ray.h"

namespace rt {

// The ONE camera model used by every renderer: pinhole, no depth of field.
struct Camera {
    Vec3 origin;
    Vec3 lower_left;   // point on the image plane (1 unit in front of origin)
    Vec3 horizontal;   // full width vector of the image plane
    Vec3 vertical;     // full height vector of the image plane

    // s, t in [0,1]; (0,0) = bottom-left of the image, (1,1) = top-right.
    RT_HD Ray generate_ray(float s, float t) const {
        Vec3 d = normalize(lower_left + horizontal * s + vertical * t - origin);
        return Ray(origin, d);
    }
};

RT_HD inline Camera make_camera(const Vec3& eye, const Vec3& target, const Vec3& up,
                                float vfov_degrees, float aspect) {
    float h  = tanf(vfov_degrees * kPi / 180.f * 0.5f);
    float vh = 2.f * h, vw = aspect * vh;
    Vec3 w = normalize(eye - target);
    Vec3 u = normalize(cross(up, w));
    Vec3 v = cross(w, u);
    Camera c;
    c.origin = eye;
    c.horizontal = u * vw;
    c.vertical = v * vh;
    c.lower_left = eye - c.horizontal * 0.5f - c.vertical * 0.5f - w;
    return c;
}

// Ray through pixel (x, y). y = 0 is the TOP row (like image files).
// jx, jy in [0,1) are the sub-pixel jitter (use 0.5, 0.5 for the pixel centre).
RT_HD inline Ray pixel_ray(const Camera& cam, int x, int y, int width, int height,
                           float jx, float jy) {
    float s = ((float)x + jx) / (float)width;
    float t = 1.f - ((float)y + jy) / (float)height;
    return cam.generate_ray(s, t);
}

}  // namespace rt
