#pragma once

#include <cstdint>

#include "../philox.h"
#include "arith.h"

// Per-counter математика monkey-workload'а. Header-only и __host__ __device__:
// один и тот же код вызывается CPU-бэкендом и CUDA-ядром, GLSL-порт — в
// src/shaders/kernels.comp. Глобальный 64-битный индекс i однозначно задаёт
// кандидата на любом бэкенде, при любом числе потоков и размере диспатча.
//
// Результат индекса — длина совпадения с эталоном (0..len): префикса для
// random, суффикса для brute. Полное совпадение — результат len.

namespace monkey {
namespace kernel {

// Индекс символа алфавита (0..n-1) для кандидата i на позиции pos. Канон
// маппинга: Philox4x32-10, counter = {pos, i_lo, i_hi, 0}, key = {seed, 0},
// символ = word0 mod n.
KERNEL_HD inline uint32_t monkey_char(uint32_t seed, uint64_t i, uint32_t pos, uint32_t n) {
    const prng::u32x4 ctr{{pos, static_cast<uint32_t>(i), static_cast<uint32_t>(i >> 32), 0u}};
    const prng::u32x2 key{{seed, 0u}};
    return prng::philox4x32_10(ctr, key).v[0] % n;
}

// random: длина совпавшего префикса кандидата i. Ранний выход на первом
// несовпадении — в среднем 1 + 1/(n-1) вызова Philox на кандидата.
KERNEL_HD inline uint32_t monkey_random_prefix(uint32_t seed, uint64_t i, const int32_t* target,
                                               uint32_t len, uint32_t n) {
    uint32_t p = 0;
    while (p < len && monkey_char(seed, i, p, n) == static_cast<uint32_t>(target[p])) ++p;
    return p;
}

// brute: кандидат — запись i в системе счисления n, дополненная ведущими
// нулями до len (младший разряд — последний символ). Результат — длина
// совпавшего суффикса; в среднем 1 + 1/(n-1) делений на кандидата.
KERNEL_HD inline uint32_t monkey_brute_suffix(uint64_t i, const int32_t* target, uint32_t len,
                                              uint32_t n) {
    uint32_t matched = 0;
    while (matched < len) {
        const uint64_t q = i / n;
        if (static_cast<uint32_t>(i - q * n) != static_cast<uint32_t>(target[len - 1 - matched]))
            break;
        i = q;
        ++matched;
    }
    return matched;
}

} // namespace kernel
} // namespace monkey
