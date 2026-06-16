#pragma once

#include <cstdint>

// Philox4x32-10 — counter-based PRNG (Salmon, Moraes, Dror, Shaw,
// "Parallel Random Numbers: As Easy as 1, 2, 3", SC'11; reference
// implementation: D.E. Shaw Research, Random123).
//
// Полностью stateless: выход определяется только (counter, key). Один и тот же
// алгоритм дословно реализован на CPU (этот заголовок), в CUDA (этот же
// заголовок под nvcc) и в GLSL (src/monkey.comp). При одинаковых входах все
// три бэкенда дают БИТ-В-БИТ идентичную последовательность — это проверяется
// тестами test_philox / test_glsl_philox против опубликованных KAT-векторов.

#if defined(__CUDACC__)
#define PRNG_HD __host__ __device__
#else
#define PRNG_HD
#endif

namespace prng {
inline namespace philox {

struct u32x4 {
    uint32_t v[4];
};
struct u32x2 {
    uint32_t v[2];
};

// Множители раундовой функции и инкременты ключа (Уэйл/Вейл-константы:
// дробные части golden ratio и sqrt(3)).
constexpr uint32_t kPhiloxM0 = 0xD2511F53u;
constexpr uint32_t kPhiloxM1 = 0xCD9E8D57u;
constexpr uint32_t kPhiloxW0 = 0x9E3779B9u;
constexpr uint32_t kPhiloxW1 = 0xBB67AE85u;

// Старшие 32 бита 64-битного произведения двух 32-битных чисел.
PRNG_HD constexpr uint32_t mulhi32(uint32_t a, uint32_t b) {
    return static_cast<uint32_t>((static_cast<uint64_t>(a) * static_cast<uint64_t>(b)) >> 32);
}

// Один раунд Philox4x32. Соответствует _philox4x32round из Random123.
PRNG_HD constexpr u32x4 philox_round(u32x4 ctr, u32x2 key) {
    const uint32_t hi0 = mulhi32(kPhiloxM0, ctr.v[0]);
    const uint32_t lo0 = kPhiloxM0 * ctr.v[0];
    const uint32_t hi1 = mulhi32(kPhiloxM1, ctr.v[2]);
    const uint32_t lo1 = kPhiloxM1 * ctr.v[2];
    return u32x4{{hi1 ^ ctr.v[1] ^ key.v[0], lo1, hi0 ^ ctr.v[3] ^ key.v[1], lo0}};
}

// Philox4x32 с 10 раундами: 1-й раунд на исходном ключе, далее ключ
// инкрементируется перед каждым последующим раундом (9 bump'ов).
PRNG_HD constexpr u32x4 philox4x32_10(u32x4 ctr, u32x2 key) {
    for (int r = 0; r < 10; ++r) {
        if (r > 0) {
            key.v[0] += kPhiloxW0;
            key.v[1] += kPhiloxW1;
        }
        ctr = philox_round(ctr, key);
    }
    return ctr;
}

// Удобная свёртка для бенчмарка: поток случайных uint32 по входу
// (seed, key, pos). Маппинг counter={key,pos,0,0}, philox-key={seed,0}
// продублирован один-в-один в GLSL (src/monkey.comp).
PRNG_HD constexpr uint32_t index_stream(uint32_t seed, uint32_t key, uint32_t pos) {
    const u32x4 ctr{{key, pos, 0u, 0u}};
    const u32x2 k{{seed, 0u}};
    return philox4x32_10(ctr, k).v[0];
}

} // namespace philox
} // namespace prng
