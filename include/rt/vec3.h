#pragma once
#include "common.h"

namespace rt {

struct Vec3 {
    float x = 0.f, y = 0.f, z = 0.f;
    Vec3() = default;
    RT_HD Vec3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}
    RT_HD explicit Vec3(float s) : x(s), y(s), z(s) {}
    RT_HD float  operator[](int i) const { return i == 0 ? x : (i == 1 ? y : z); }
    RT_HD float& operator[](int i)       { return i == 0 ? x : (i == 1 ? y : z); }
    RT_HD Vec3 operator-() const { return Vec3(-x, -y, -z); }
    RT_HD Vec3& operator+=(const Vec3& o) { x += o.x; y += o.y; z += o.z; return *this; }
    RT_HD Vec3& operator-=(const Vec3& o) { x -= o.x; y -= o.y; z -= o.z; return *this; }
    RT_HD Vec3& operator*=(float s)       { x *= s; y *= s; z *= s; return *this; }
    RT_HD Vec3& operator*=(const Vec3& o) { x *= o.x; y *= o.y; z *= o.z; return *this; }
};

RT_HD inline Vec3 operator+(Vec3 a, const Vec3& b) { a += b; return a; }
RT_HD inline Vec3 operator-(Vec3 a, const Vec3& b) { a -= b; return a; }
RT_HD inline Vec3 operator*(Vec3 a, float s)       { a *= s; return a; }
RT_HD inline Vec3 operator*(float s, Vec3 a)       { a *= s; return a; }
RT_HD inline Vec3 operator*(Vec3 a, const Vec3& b) { a *= b; return a; }   // component-wise
RT_HD inline Vec3 operator/(Vec3 a, float s)       { float r = 1.f / s; a *= r; return a; }

RT_HD inline float dot(const Vec3& a, const Vec3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
RT_HD inline Vec3 cross(const Vec3& a, const Vec3& b) {
    return Vec3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}
RT_HD inline float length2(const Vec3& a) { return dot(a, a); }
RT_HD inline float length(const Vec3& a)  { return sqrtf(dot(a, a)); }
RT_HD inline Vec3 normalize(const Vec3& a) {
    float l = length(a);
    return l > 0.f ? a / l : Vec3(0.f, 0.f, 0.f);
}
RT_HD inline Vec3 vmin(const Vec3& a, const Vec3& b) { return Vec3(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z)); }
RT_HD inline Vec3 vmax(const Vec3& a, const Vec3& b) { return Vec3(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z)); }
RT_HD inline Vec3 lerp(const Vec3& a, const Vec3& b, float t) { return a * (1.f - t) + b * t; }
RT_HD inline float clampf(float v, float lo, float hi) { return fminf(fmaxf(v, lo), hi); }
RT_HD inline Vec3 reflect(const Vec3& v, const Vec3& n) { return v - n * (2.f * dot(v, n)); }
RT_HD inline float max_component(const Vec3& a) { return fmaxf(a.x, fmaxf(a.y, a.z)); }

}  // namespace rt
