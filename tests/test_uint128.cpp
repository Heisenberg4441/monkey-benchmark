// Unit-тесты portable Counter (128 бит): overflow boundary, арифметика для
// brute-индекса. Использует только API counter_from_u64 / counter_lo /
// counter_hi / counter_divmod — никаких предположений о внутреннем устройстве
// Counter (на GCC/Clang это __uint128_t, в CUDA/на MSVC — struct).
#include "catch.hpp"
#include "uint128.h"

#include <cstdint>

using monkey::Counter;
using monkey::counter_divmod;
using monkey::counter_from_u64;
using monkey::counter_hi;
using monkey::counter_lo;

TEST_CASE("counter_from_u64 roundtrips", "[uint128]") {
    const uint64_t values[] = {0ULL, 1ULL, UINT64_MAX, UINT32_MAX, 0xDEADBEEFCAFEBABEULL};
    for (auto v : values) {
        Counter c = counter_from_u64(v);
        REQUIRE(counter_lo(c) == v);
        REQUIRE(counter_hi(c) == 0);
    }
}

TEST_CASE("Counter add matches 128-bit arithmetic", "[uint128]") {
    SECTION("lo-only, no carry") {
        Counter a = counter_from_u64(42);
        Counter b = counter_from_u64(100);
        Counter r = a + b;
        REQUIRE(counter_lo(r) == 142);
        REQUIRE(counter_hi(r) == 0);
    }
    SECTION("carry into hi") {
        Counter a = counter_from_u64(UINT64_MAX);
        Counter b = counter_from_u64(1);
        Counter r = a + b;
        REQUIRE(counter_lo(r) == 0);
        REQUIRE(counter_hi(r) == 1);
    }
    SECTION("hi carry propagation") {
        // UINT64_MAX + 1 wraps lo to 0 and carries into hi.
        Counter a = counter_from_u64(UINT64_MAX);
        Counter r = a + counter_from_u64(1);
        REQUIRE(counter_lo(r) == 0);
        REQUIRE(counter_hi(r) == 1);
        // Add another UINT64_MAX: lo becomes UINT64_MAX, hi stays 1.
        Counter r2 = r + counter_from_u64(UINT64_MAX);
        REQUIRE(counter_lo(r2) == UINT64_MAX);
        REQUIRE(counter_hi(r2) == 1);
    }
}

TEST_CASE("Counter multiply by uint64_t", "[uint128]") {
    SECTION("zero * anything = zero") {
        Counter r = counter_from_u64(0) * 12345ULL;
        REQUIRE(counter_lo(r) == 0);
        REQUIRE(counter_hi(r) == 0);
    }
    SECTION("small * small") {
        Counter r = counter_from_u64(1024) * 1048576ULL;
        REQUIRE(counter_lo(r) == 1024ULL * 1048576ULL);
        REQUIRE(counter_hi(r) == 0);
    }
    SECTION("2^64-1 * 2 has correct hi") {
        Counter a = counter_from_u64(UINT64_MAX);
        Counter r = a * 2ULL;
        REQUIRE(counter_lo(r) == UINT64_MAX - 1);
        REQUIRE(counter_hi(r) == 1);
    }
    SECTION("2^64 * 2^62 gives hi = 2^62, lo = 0") {
        // 2^64 expressed as 2^32 * 2^32
        Counter a = counter_from_u64(0x100000000ULL) * 0x100000000ULL;
        Counter r = a * (1ULL << 62);
        REQUIRE(counter_lo(r) == 0);
        REQUIRE(counter_hi(r) == (1ULL << 62));
        // Verify a = 2^64
        REQUIRE(counter_lo(a) == 0);
        REQUIRE(counter_hi(a) == 1);
    }
}

TEST_CASE("counter_divmod by uint32_t", "[uint128]") {
    SECTION("fits in lo") {
        Counter n = counter_from_u64(1000);
        uint32_t rem = 0;
        Counter q = counter_divmod(n, 7, &rem);
        REQUIRE(counter_lo(q) == 142); // 1000/7
        REQUIRE(counter_hi(q) == 0);
        REQUIRE(rem == 6);
    }
    SECTION("hi > 0, exact division") {
        // n = 2^64
        Counter n = counter_from_u64(0x100000000ULL) * 0x100000000ULL;
        uint32_t rem = 0;
        Counter q = counter_divmod(n, 2, &rem);
        // q = 2^63
        REQUIRE(counter_lo(q) == 0x8000000000000000ULL);
        REQUIRE(counter_hi(q) == 0);
        REQUIRE(rem == 0);
    }
    SECTION("hi > 0, non-exact") {
        // n = 2^64 + 1
        Counter n = counter_from_u64(0x100000000ULL) * 0x100000000ULL +
                    counter_from_u64(1);
        uint32_t rem = 0;
        Counter q = counter_divmod(n, 3, &rem);
        // 2^64 + 1 = 18446744073709551617, /3 = 6148914691236517205 R2
        REQUIRE(counter_lo(q) == 6148914691236517205ULL);
        REQUIRE(counter_hi(q) == 0);
        REQUIRE(rem == 2);
    }
}

TEST_CASE("N=26 L=20 overflow boundary", "[uint128][brute]") {
    // 26^20 ≈ 2^94 — не влезает в 64 бита, влезает в 128.
    Counter max_val = counter_from_u64(1);
    for (int i = 0; i < 20; ++i) max_val = max_val * 26ULL;

    REQUIRE(counter_hi(max_val) != 0); // upper bits populated
    REQUIRE(counter_lo(max_val) != 0);

    // Кодируем 26^20 - 1 вручную: 20 цифр по 25 дают последний индекс.
    Counter last = counter_from_u64(0);
    for (int i = 0; i < 20; ++i) last = last * 26ULL + counter_from_u64(25);

    // Извлекаем цифры через divmod — должны получить 25, 25, ..., 25.
    Counter t = last;
    for (int step = 0; step < 20; ++step) {
        uint32_t rem = 0;
        t = counter_divmod(t, 26, &rem);
        REQUIRE(rem == 25);
    }
    REQUIRE(counter_lo(t) == 0);
    REQUIRE(counter_hi(t) == 0);
}

TEST_CASE("Counter enumeration cycle preserves total", "[uint128]") {
    // Проверка: если просуммировать 10^6 последовательных Counter-ов,
    // разница между последним и первым должна быть 10^6 - 1.
    constexpr uint64_t kN = 1'000'000;
    Counter a = counter_from_u64(0xABCD);
    Counter b = a;
    for (uint64_t i = 1; i < kN; ++i) b = b + counter_from_u64(1);
    Counter expected = a + counter_from_u64(kN - 1);
    REQUIRE(counter_lo(b) == counter_lo(expected));
    REQUIRE(counter_hi(b) == counter_hi(expected));
}
