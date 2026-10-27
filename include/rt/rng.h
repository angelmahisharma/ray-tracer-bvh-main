#pragma once
#include "common.h"

namespace rt {

// PCG32 random generator. Same code on CPU, CUDA and OptiX so every renderer
// draws from the same kind of random stream.
//
// Convention (see docs/HOW_TO_PLUG_IN.md):
//     Rng rng;  rng.seed(global_seed, pixel_index);   // one stream per pixel
struct Rng {
    uint64_t state = 0x853c49e6748fea9bULL;
    uint64_t inc   = 0xda3e39cb94b95bdbULL;

    RT_HD uint32_t next_u32() {
        uint64_t old = state;
        state = old * 6364136223846793005ULL + inc;
        uint32_t xorshifted = (uint32_t)(((old >> 18u) ^ old) >> 27u);
        uint32_t rot = (uint32_t)(old >> 59u);
        return (xorshifted >> rot) | (xorshifted << ((~rot + 1u) & 31u));
    }
    RT_HD void seed(uint64_t init_state, uint64_t stream) {
        state = 0u;
        inc = (stream << 1u) | 1u;
        next_u32();
        state += init_state;
        next_u32();
    }
    // Uniform float in [0, 1).
    RT_HD float next_float() { return (float)(next_u32() >> 8) * (1.0f / 16777216.0f); }
};

}  // namespace rt
