// Проверка детерминированного Miller-Rabin: известные простые и составные,
// полный перебор против решета и окна на больших n против сегментированного решета.
#include "catch.hpp"
#include "sieve.h"
#include "workloads/miller_rabin_kernel.h"

#include <cstdint>

using monkey::kernel::miller_rabin_is_prime;

TEST_CASE("Miller-Rabin deterministic for known primes", "[millerrabin][kernel]") {
    SECTION("small primes") {
        const uint64_t primes[] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37,
                                   41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83,
                                   89, 97, 101, 313, 997};
        for (auto p : primes) {
            INFO("prime " << p);
            REQUIRE(miller_rabin_is_prime(p));
        }
    }

    SECTION("small composites") {
        const uint64_t comp[] = {0, 1, 4, 6, 8, 9, 10, 12, 14, 15, 16, 18, 20,
                                 21, 22, 24, 25, 26, 27, 28, 30, 100, 102, 105};
        for (auto c : comp) {
            INFO("composite " << c);
            REQUIRE_FALSE(miller_rabin_is_prime(c));
        }
    }

    SECTION("large known primes (> 2^32)") {
        REQUIRE(miller_rabin_is_prime(1000000007ULL));
        REQUIRE(miller_rabin_is_prime(1099511627689ULL));
        REQUIRE(miller_rabin_is_prime(3314192745739ULL));
        REQUIRE(miller_rabin_is_prime(18446744073709551557ULL)); // наибольшее < 2^64
    }

    SECTION("large composites") {
        REQUIRE_FALSE(miller_rabin_is_prime(1000000008ULL));
        REQUIRE_FALSE(miller_rabin_is_prime(1099511627695ULL));
        REQUIRE_FALSE(miller_rabin_is_prime(3314192745741ULL));
        REQUIRE_FALSE(miller_rabin_is_prime(18446744073709551615ULL)); // 2^64 - 1
        // Сильное псевдопростое по всем основаниям 2..37 (Jaeschke).
        REQUIRE_FALSE(miller_rabin_is_prime(3825123056546413051ULL));
    }

    SECTION("strong pseudoprimes to bases {2, 325} are rejected") {
        // Регрессия: при «a >= n => пропуск» эти числа проходили только по 2 и 325.
        REQUIRE_FALSE(miller_rabin_is_prime(4033));
        REQUIRE_FALSE(miller_rabin_is_prime(4681));
    }
}

TEST_CASE("Miller-Rabin matches the sieve for every n < 2^20", "[millerrabin][sieve]") {
    constexpr uint64_t kLimit = 1u << 20;
    const auto is_prime = test_sieve::simple(kLimit);
    uint64_t mismatches = 0, first = 0;
    for (uint64_t n = 0; n < kLimit; ++n) {
        if (miller_rabin_is_prime(n) != (is_prime[n] != 0)) {
            if (mismatches++ == 0) first = n;
        }
    }
    INFO("first mismatch at n = " << first);
    REQUIRE(mismatches == 0);
}

TEST_CASE("Miller-Rabin matches a segmented sieve on large windows",
          "[millerrabin][sieve]") {
    // Окна около 2^32, 2^40, 2^48: базовые простые до 2^24 — решето за доли секунды.
    for (uint64_t lo : {(1ull << 32) - 50000, 1ull << 40, (1ull << 48) - 60000}) {
        const uint64_t hi = lo + 100000;
        const auto is_prime = test_sieve::segment(lo, hi);
        uint64_t mismatches = 0, primes = 0;
        for (uint64_t n = lo; n < hi; ++n) {
            const bool truth = is_prime[n - lo] != 0;
            primes += truth;
            if (miller_rabin_is_prime(n) != truth) ++mismatches;
        }
        INFO("window [" << lo << ", " << hi << ")");
        REQUIRE(primes > 0);
        REQUIRE(mismatches == 0);
    }
}

namespace {

// Пошаговый автомат (вариант для GPU) до итога.
bool lane_is_prime(uint64_t n) {
    monkey::kernel::MrLane L{};
    int r = monkey::kernel::mr_lane_start(L, n);
    while (r == monkey::kernel::kMrContinue) r = monkey::kernel::mr_lane_step(L);
    return r == monkey::kernel::kMrPrime;
}

} // namespace

TEST_CASE("step-wise SIMT Miller-Rabin equals the scalar kernel", "[millerrabin][simt]") {
    uint64_t mismatches = 0;
    for (uint64_t n = 0; n < (1u << 18); ++n) mismatches += lane_is_prime(n) != miller_rabin_is_prime(n);
    // d = 1 (n = 2^s + 1), сильные псевдопростые и края 64 бит.
    for (uint64_t n : {5ull, 17ull, 257ull, 65537ull, 4294967297ull, 4033ull, 4681ull,
                       3825123056546413051ull, 18446744073709551557ull, ~0ull, (1ull << 63) + 1})
        mismatches += lane_is_prime(n) != miller_rabin_is_prime(n);
    for (uint64_t lo : {1ull << 40, (1ull << 62) + 1, ~0ull - 200000}) {
        for (uint64_t n = lo; n < lo + 100000; ++n) mismatches += lane_is_prime(n) != miller_rabin_is_prime(n);
    }
    REQUIRE(mismatches == 0);
}
