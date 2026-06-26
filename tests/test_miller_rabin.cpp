// Проверка детерминированного Miller-Rabin: известные простые и составные.
#include "catch.hpp"
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
    }

    SECTION("large composites") {
        REQUIRE_FALSE(miller_rabin_is_prime(1000000008ULL));
        REQUIRE_FALSE(miller_rabin_is_prime(1099511627695ULL));
        REQUIRE_FALSE(miller_rabin_is_prime(3314192745741ULL));
    }

    SECTION("deterministic repeatability") {
        for (uint64_t n : {313ULL, 1000000007ULL}) {
            REQUIRE(miller_rabin_is_prime(n) == miller_rabin_is_prime(n));
        }
    }
}
