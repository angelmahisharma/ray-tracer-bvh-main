// Shared basics. Everything in include/rt/ that is marked RT_HD can be used
// from plain C++, from CUDA (.cu) code and from OptiX device programs.
#pragma once
#include <cstdint>
#include <math.h>

#if defined(__CUDACC__)
#define RT_HD __host__ __device__
#else
#define RT_HD
#endif

namespace rt {
constexpr float kPi        = 3.14159265358979323846f;
constexpr float kBigFloat  = 1.0e30f;   // "infinity" that is safe on GPU
constexpr float kRayEps    = 1.0e-4f;   // default ray tmin (avoid self-hit)
}  // namespace rt
