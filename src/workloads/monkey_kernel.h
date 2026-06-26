#pragma once

#include <cstdint>

#include "../philox.h"

// Per-counter математика monkey-workload'а. Header-only и __host__ __device__:
// один и тот же код вызывается CPU-workload'ом и CUDA-ядром, поэтому кандидат
// для данного counter'а одинаков на всех бэкендах.

#if defined(__CUDACC__)
#define WK_HD __host__ __device__
#else
#define WK_HD
#endif

namespace monkey {
namespace kernel {

// Индекс символа алфавита (0..n-1) для кандидата с глобальным counter на
// позиции pos. Полностью counter-based: counter -> 64-битные слова Philox.
WK_HD inline int monkey_char(uint32_t seed, uint64_t counter, uint32_t pos, int n) {
    const prng::u32x4 ctr{{pos, static_cast<uint32_t>(counter),
                           static_cast<uint32_t>(counter >> 32), 0u}};
    const prng::u32x2 key{{seed, 0u}};
    return static_cast<int>(prng::philox4x32_10(ctr, key).v[0] % static_cast<uint32_t>(n));
}

// Совпадает ли кандидат counter'а с эталоном target[0..len)?
// mode 0 = random (кандидат из PRNG), 1 = brute (counter как число по основанию n).
WK_HD inline bool monkey_match(int mode, uint32_t seed, uint64_t counter,
                               const int* target, int len, int n) {
    if (mode == 0) {
        for (int p = 0; p < len; ++p) {
            if (monkey_char(seed, counter, static_cast<uint32_t>(p), n) != target[p]) {
                return false;
            }
        }
        return true;
    }
    // brute: младший разряд — последний символ.
    uint64_t t = counter;
    const uint32_t base = static_cast<uint32_t>(n);
    for (int p = len; p-- > 0;) {
        const int digit = static_cast<int>(t % base);
        t /= base;
        if (digit != target[p]) {
            return false;
        }
    }
    return true;
}

} // namespace kernel
} // namespace monkey
