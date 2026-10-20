#pragma once

#include <cstdint>

#include "arith.h"

// Miller-Rabin deterministic primality test for 64-bit integers.
// Uses known witnesses for n < 2^64 per Jim Sinclair (2011):
//   n < 2^64: {2, 325, 9375, 28178, 450775, 9780504, 1795265022}
// Stateless: результат зависит только от n (counter'а), без PRNG.
//
// Арифметика — Montgomery с R = 2^64 (arith.h): ни одного 128-битного деления
// в горячем пути, одинаково на CPU, CUDA и GLSL (src/shaders/kernels.comp).

namespace monkey {
namespace kernel {

// Montgomery-контекст одного нечётного n: n^-1 mod 2^64 и представления 1 и -1.
struct MrCtx {
    uint64_t n;
    uint64_t ninv;
    uint64_t one;  // R mod n
    uint64_t mone; // (n - 1)·R mod n
};

KERNEL_HD inline uint64_t mr_mul(uint64_t a, uint64_t b, const MrCtx& c) {
    return mont64_mul(a, b, c.n, c.ninv);
}

// base^e для base в форме Монтгомери (справа налево).
KERNEL_HD inline uint64_t mr_pow(uint64_t base, uint64_t e, const MrCtx& c) {
    uint64_t r = c.one;
    while (e) {
        if (e & 1u) r = mr_mul(r, base, c);
        e >>= 1u;
        if (e) base = mr_mul(base, base, c);
    }
    return r;
}

// 2^e слева направо: умножение на основание 2 — это модульное удвоение, а не
// mont-умножение. Для составных обычно проверяется только свидетель 2, поэтому
// это основной путь workload'а.
KERNEL_HD inline uint64_t mr_pow2(uint64_t e, const MrCtx& c) {
    uint64_t r = c.one;
    for (int i = 63 - clz64(e); i >= 0; --i) {
        r = mr_mul(r, r, c);
        if ((e >> i) & 1u) r = addmod64(r, r, c.n);
    }
    return r;
}

// Сильная проверка по x = a^d (форма Монтгомери), n - 1 = d·2^s.
KERNEL_HD inline bool mr_strong(uint64_t x, int s, const MrCtx& c) {
    if (x == c.one || x == c.mone) return true;
    for (int r = 1; r < s; ++r) {
        x = mr_mul(x, x, c);
        if (x == c.mone) return true;
        if (x == c.one) return false;
    }
    return false;
}

// Детерминированный тест простоты для n < 2^64.
KERNEL_HD inline bool miller_rabin_is_prime(uint64_t n) {
    if (n < 2) return false;
    if (n < 4) return true;
    if ((n & 1u) == 0) return false;

    // n-1 = d * 2^s, d — нечётное.
    const int s = ctz64(n - 1);
    const uint64_t d = (n - 1) >> s;

    MrCtx c;
    c.n = n;
    c.ninv = inv64(n);
    c.one = pow2_64_mod(n);
    c.mone = n - c.one;

    // Свидетели — каждый берётся mod n, нулевой пропускается. Пропуск a >= n
    // (вместо редукции) давал ложные простые: 4033 и 4681 проходят по {2, 325}.
    if (!mr_strong(mr_pow2(d, c), s, c)) return false;

    // R^2 mod n — удвоением R mod n ещё 64 раза; нужен только тем n, что
    // прошли свидетеля 2 (простые и редкие псевдопростые).
    uint64_t r2 = c.one;
    for (int i = 0; i < 64; ++i) r2 = addmod64(r2, r2, n);

    const uint64_t witnesses[] = {325, 9375, 28178, 450775, 9780504, 1795265022};
    for (uint64_t a : witnesses) {
        a %= n;
        if (a == 0) continue;
        if (!mr_strong(mr_pow(mr_mul(a, r2, c), d, c), s, c)) return false;
    }
    return true;
}

} // namespace kernel
} // namespace monkey
