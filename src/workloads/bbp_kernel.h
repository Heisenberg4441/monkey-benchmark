#pragma once

#include <cmath>
#include <cstdint>

// BBP kernel: вычисление k-го hex-знака числа π по формуле
// Bailey-Borwein-Plouffe (1996). Полностью stateless — только от counter'а.
//
// Арифметика: double + модулярное возведение (powmod16). На CUDA double
// эквивалентна IEEE 754 — результаты бит-в-бит совпадают с CPU.
// verify() сверяет первые 1024 знака с reference/pi_hex.txt.

#if defined(__CUDACC__)
#define BBP_HD __host__ __device__
#else
#define BBP_HD
#endif

namespace monkey {
namespace kernel {

// 16^e mod m через fast exponentiation.
BBP_HD inline uint64_t bbp_powmod16(uint64_t e, uint64_t m) {
    if (m == 0) return 0;
    uint64_t r = 1;
    uint64_t b = 16 % m;
    while (e) {
        if (e & 1u)
            r = static_cast<uint64_t>((static_cast<__uint128_t>(r) * b) % m);
        b = static_cast<uint64_t>((static_cast<__uint128_t>(b) * b) % m);
        e >>= 1u;
    }
    return r;
}

// BBP S_j(d): sum_{k=0..∞} (16^{d-k} mod (8k+j)) / (8k+j) по mod 1.
BBP_HD inline double bbp_series_mod1(uint64_t d, uint32_t j) {
    double s = 0.0;

    // Part 1: k = 0..d — доминирующая (O(d) итераций).
    for (uint64_t k = 0; k <= d; ++k) {
        const uint64_t denom = 8ULL * k + j;
        const uint64_t num = bbp_powmod16(d - k, denom);
        s += static_cast<double>(num) / static_cast<double>(denom);
        s -= floor(s);
    }

    // Part 2: k = d+1..∞ — быстро затухает, ~15 членов хватает.
    double pow16 = 1.0 / 16.0;
    for (uint64_t k = d + 1; k <= d + 20; ++k) {
        const double term = pow16 / static_cast<double>(8ULL * k + j);
        s += term;
        if (term < 1e-17) break;
        pow16 /= 16.0;
    }
    return s - floor(s);
}

// Hex-цифра π на позиции d (d=0 — первый знак после 3.).
// Возвращает 0..15.
BBP_HD inline int bbp_hex_digit(uint64_t d) {
    double frac = 4.0 * bbp_series_mod1(d, 1) - 2.0 * bbp_series_mod1(d, 4) -
                  bbp_series_mod1(d, 5) - bbp_series_mod1(d, 6);
    frac = frac - floor(frac);
    if (frac < 0.0) frac += 1.0;
    int dig = static_cast<int>(frac * 16.0);
    return (dig < 0) ? 0 : ((dig > 15) ? 15 : dig);
}

} // namespace kernel
} // namespace monkey

#undef BBP_HD
