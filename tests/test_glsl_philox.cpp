// Слой 1 проверки GLSL-бэкенда: дословная C++ транслитерация Philox из
// src/monkey.comp. Сверяется с каноном (philox.h) на первых 2^16 выходах.
// Это ловит опечатку в переносе шейдера без запуска Vulkan. Слой 2 — реальный
// прогон скомпилированного SPIR-V — в test_vulkan_philox.cpp (если есть устройство).
#include "catch.hpp"
#include "philox.h"

#include <cstdint>

namespace glsl_mirror {

// Эмуляция GLSL umulExtended(x, y, out msb, out lsb): msb — старшие 32 бита.
inline void umulExtended(uint32_t x, uint32_t y, uint32_t& msb, uint32_t& lsb) {
    const uint64_t p = static_cast<uint64_t>(x) * static_cast<uint64_t>(y);
    msb = static_cast<uint32_t>(p >> 32);
    lsb = static_cast<uint32_t>(p);
}

struct uvec4 {
    uint32_t x, y, z, w;
};
struct uvec2 {
    uint32_t x, y;
};

constexpr uint32_t PHILOX_M0 = 0xD2511F53u;
constexpr uint32_t PHILOX_M1 = 0xCD9E8D57u;
constexpr uint32_t PHILOX_W0 = 0x9E3779B9u;
constexpr uint32_t PHILOX_W1 = 0xBB67AE85u;

inline uvec4 philox_round(uvec4 ctr, uvec2 key) {
    uint32_t hi0, lo0, hi1, lo1;
    umulExtended(PHILOX_M0, ctr.x, hi0, lo0);
    umulExtended(PHILOX_M1, ctr.z, hi1, lo1);
    return uvec4{hi1 ^ ctr.y ^ key.x, lo1, hi0 ^ ctr.w ^ key.y, lo0};
}

inline uvec4 philox4x32_10(uvec4 ctr, uvec2 key) {
    for (int r = 0; r < 10; ++r) {
        if (r > 0) {
            key.x += PHILOX_W0;
            key.y += PHILOX_W1;
        }
        ctr = philox_round(ctr, key);
    }
    return ctr;
}

inline uint32_t index_stream(uint32_t seed, uint32_t key, uint32_t pos) {
    return philox4x32_10(uvec4{key, pos, 0u, 0u}, uvec2{seed, 0u}).x;
}

} // namespace glsl_mirror

TEST_CASE("GLSL Philox transliteration matches CPU over 2^16 outputs", "[glsl][determinism]") {
    constexpr uint32_t kSeed = 0x9e3779b9u;
    constexpr uint32_t kN = 1u << 16;
    bool all_equal = true;
    for (uint32_t i = 0; i < kN; ++i) {
        const uint32_t key = i >> 4;       // покрываем разные key...
        const uint32_t pos = i & 0xFu;     // ...и pos
        if (glsl_mirror::index_stream(kSeed, key, pos) !=
            prng::index_stream(kSeed, key, pos)) {
            all_equal = false;
            break;
        }
    }
    REQUIRE(all_equal);
}
