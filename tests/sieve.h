#pragma once

// Независимый эталон простоты для тестов: решето Эратосфена и сегментированное
// решето окна [lo, hi). Ничего общего с ядром Miller-Rabin не имеет — только
// деление по простым до sqrt(hi).

#include <cmath>
#include <cstdint>
#include <vector>

namespace test_sieve {

// is_prime[i] для i < limit.
inline std::vector<uint8_t> simple(uint64_t limit) {
    std::vector<uint8_t> p(limit, 1);
    for (uint64_t i = 0; i < limit && i < 2; ++i) p[i] = 0;
    for (uint64_t i = 2; i * i < limit; ++i)
        if (p[i])
            for (uint64_t j = i * i; j < limit; j += i) p[j] = 0;
    return p;
}

// Целочисленный floor(sqrt(x)) без погрешностей double на больших x.
inline uint64_t isqrt(uint64_t x) {
    uint64_t r = static_cast<uint64_t>(std::sqrt(static_cast<double>(x)));
    while (r > 0 && r * r > x) --r;
    while ((r + 1) * (r + 1) <= x) ++r;
    return r;
}

// is_prime[i] для числа lo + i, i < hi - lo. Требует lo >= 2.
inline std::vector<uint8_t> segment(uint64_t lo, uint64_t hi) {
    std::vector<uint8_t> seg(hi - lo, 1);
    const uint64_t root = isqrt(hi - 1);
    const std::vector<uint8_t> base = simple(root + 1);
    for (uint64_t p = 2; p <= root; ++p) {
        if (!base[p]) continue;
        uint64_t start = (lo + p - 1) / p * p;
        if (start < p * p) start = p * p;
        for (uint64_t m = start; m < hi; m += p) seg[m - lo] = 0;
    }
    return seg;
}

} // namespace test_sieve
