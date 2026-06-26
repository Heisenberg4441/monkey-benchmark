#pragma once

#include <cstdint>

// Portable 128-bit unsigned integer for brute-force counter arithmetic.
// On GCC/Clang host (where __uint128_t is available) the native type is used
// as Counter; on MSVC and in CUDA __device__ code a custom struct provides
// the same semantics with add / mul-u64 / divmod-u32.
//
// The interface is deliberately narrow: only the operations needed by the
// monkey-brute hot loop.  This keeps the CUDA device port trivial.

#if defined(__SIZEOF_INT128__) && !defined(__CUDACC__) && !defined(_MSC_VER)

namespace monkey {

using Counter = __uint128_t;

inline constexpr Counter counter_from_u64(uint64_t v) { return v; }
inline constexpr uint64_t counter_lo(Counter c) { return static_cast<uint64_t>(c); }
inline constexpr uint64_t counter_hi(Counter c) { return static_cast<uint64_t>(c >> 64); }
inline constexpr bool counter_eq_zero(Counter c) { return c == 0; }

inline Counter counter_divmod(Counter n, uint32_t d, uint32_t* rem) {
    const Counter q = n / d;
    *rem = static_cast<uint32_t>(n % d);
    return q;
}

} // namespace monkey

#else

#if defined(__CUDACC__)
#define U128_HD __host__ __device__
#else
#define U128_HD
#endif

namespace monkey {

struct Counter {
    uint64_t lo;
    uint64_t hi;
};

U128_HD inline Counter counter_from_u64(uint64_t v) { return {v, 0}; }
U128_HD inline uint64_t counter_lo(Counter c) { return c.lo; }
U128_HD inline uint64_t counter_hi(Counter c) { return c.hi; }

U128_HD inline bool counter_eq_zero(Counter c) { return c.lo == 0 && c.hi == 0; }
U128_HD inline bool operator==(Counter a, Counter b) { return a.lo == b.lo && a.hi == b.hi; }
U128_HD inline bool operator!=(Counter a, Counter b) { return !(a == b); }

U128_HD inline Counter operator+(Counter a, Counter b) {
    const uint64_t lo = a.lo + b.lo;
    return {lo, a.hi + b.hi + (lo < a.lo ? 1u : 0u)};
}

U128_HD inline Counter& operator+=(Counter& a, Counter b) { a = a + b; return a; }

// Multiply Counter by uint64_t (result fits in 128 bits because
// Counter is already < 2^128 and the second operand is < 2^64).
U128_HD inline Counter operator*(Counter a, uint64_t b) {
    // Splat: a = a_hi*2^64 + a_lo,  a*b = a_hi*b*2^64 + a_lo*b
    const uint64_t lo_lo = static_cast<uint64_t>(static_cast<uint32_t>(a.lo)) *
                           static_cast<uint32_t>(b);
    const uint64_t lo_hi =
        static_cast<uint64_t>(static_cast<uint32_t>(a.lo >> 32)) * static_cast<uint32_t>(b);
    const uint64_t hi_lo =
        static_cast<uint64_t>(static_cast<uint32_t>(a.hi)) * static_cast<uint32_t>(b);
    const uint64_t hi_hi =
        static_cast<uint64_t>(static_cast<uint32_t>(a.hi >> 32)) * static_cast<uint32_t>(b);

    const uint64_t mid0 = lo_hi + (lo_lo >> 32);
    const uint64_t mid1 = hi_lo + static_cast<uint32_t>(mid0);
    const uint64_t hi = hi_hi + (mid0 >> 32) + (mid1 >> 32);
    const uint64_t lo = (static_cast<uint32_t>(lo_lo)) | (mid1 << 32);
    return {lo, hi};
}

U128_HD inline Counter operator*(uint64_t a, Counter b) { return b * a; }

// Narrow division: Counter / uint32_t, Counter % uint32_t.
// Returns quotient; *rem = remainder.
U128_HD inline Counter counter_divmod(Counter n, uint32_t d, uint32_t* rem) {
    if (n.hi == 0) {
        const uint64_t q = n.lo / d;
        *rem = static_cast<uint32_t>(n.lo % d);
        return {q, 0};
    }
    // 4-word / 1-word schoolbook division, MSW-first.
    uint32_t w[4] = {
        static_cast<uint32_t>(n.lo),
        static_cast<uint32_t>(n.lo >> 32),
        static_cast<uint32_t>(n.hi),
        static_cast<uint32_t>(n.hi >> 32),
    };
    uint64_t r = 0;
    for (int i = 3; i >= 0; --i) {
        const uint64_t cur = (r << 32) | w[i];
        w[i] = static_cast<uint32_t>(cur / d);
        r = cur % d;
    }
    *rem = static_cast<uint32_t>(r);
    return {{static_cast<uint64_t>(w[0]) | (static_cast<uint64_t>(w[1]) << 32),
             static_cast<uint64_t>(w[2]) | (static_cast<uint64_t>(w[3]) << 32)}};
}

} // namespace monkey

#undef U128_HD
#endif
