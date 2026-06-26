#pragma once

#include <cstdint>

// Miller-Rabin deterministic primality test for 64-bit integers.
// Uses known witnesses for n < 2^64 per Jim Sinclair (2011):
//   n < 2^64: {2, 325, 9375, 28178, 450775, 9780504, 1795265022}
// Stateless: результат зависит только от n (counter'а), без PRNG.

#if defined(__CUDACC__)
#define MR_HD __host__ __device__
#else
#define MR_HD
#endif

namespace monkey {
namespace kernel {

// (a * b) % m без 128-битного переполнения.
// На GCC/Clang-хосте использует родной __uint128_t; на остальных — бинарный метод.
MR_HD inline uint64_t mr_modmul(uint64_t a, uint64_t b, uint64_t m) {
#if defined(__SIZEOF_INT128__) && !defined(__CUDACC__) && !defined(_MSC_VER)
    return static_cast<uint64_t>((static_cast<__uint128_t>(a) * b) % m);
#else
    if (m == 0) return 0;
    uint64_t r = 0;
    a %= m;
    while (b) {
        if (b & 1u) {
            r += a;
            if (r >= m) r -= m;
        }
        a <<= 1u;
        if (a >= m) a -= m;
        b >>= 1u;
    }
    return r;
#endif
}

// (a ^ e) % m через fast exponentiation с modmul.
MR_HD inline uint64_t mr_powermod(uint64_t a, uint64_t e, uint64_t m) {
    uint64_t r = 1;
    a %= m;
    while (e) {
        if (e & 1u) r = mr_modmul(r, a, m);
        a = mr_modmul(a, a, m);
        e >>= 1u;
    }
    return r;
}

// Miller-Rabin для одного свидетеля a.
MR_HD inline bool mr_witness(uint64_t a, uint64_t d, int s, uint64_t n) {
    uint64_t x = mr_powermod(a, d, n);
    if (x == 1 || x == n - 1) return true;
    for (int r = 1; r < s; ++r) {
        x = mr_modmul(x, x, n);
        if (x == n - 1) return true;
        if (x == 1) return false;
    }
    return false;
}

// Детерминированный тест простоты для n < 2^64.
MR_HD inline bool miller_rabin_is_prime(uint64_t n) {
    if (n < 2) return false;
    if (n == 2 || n == 3) return true;
    if ((n & 1u) == 0) return false;

    // n-1 = d * 2^s, d — нечётное.
    uint64_t d = n - 1;
    int s = 0;
    while ((d & 1u) == 0) {
        d >>= 1u;
        ++s;
    }

    // Детерминированные свидетели для n < 2^64 (Jim Sinclair).
    const uint64_t witnesses[] = {2, 325, 9375, 28178, 450775, 9780504, 1795265022};
    for (auto a : witnesses) {
        if (a >= n) continue;
        if (!mr_witness(a, d, s, n)) return false;
    }
    return true;
}

} // namespace kernel
} // namespace monkey

#undef MR_HD
