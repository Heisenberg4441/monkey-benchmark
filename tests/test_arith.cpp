// Общая арифметика ядер (workloads/arith.h): fmix64 против опубликованного
// splitmix64, mulhi64 и Montgomery против независимого эталона. Эталон —
// побайтно-безопасное сложение с удвоением, без 128-битных типов, поэтому
// тест одинаково строг и на MSVC, и в сборке MONKEY_PORTABLE_ARITH.
#include "catch.hpp"
#include "workloads/arith.h"

#include <cstdint>
#include <random>

using namespace monkey::kernel;

namespace {

// (a·b) mod m сложением с удвоением; addmod не переполняется ни при каком m.
uint64_t ref_mulmod(uint64_t a, uint64_t b, uint64_t m) {
    a %= m;
    uint64_t r = 0;
    while (b) {
        if (b & 1) r = addmod64(r, a, m);
        a = addmod64(a, a, m);
        b >>= 1;
    }
    return r;
}

// Старшие 64 бита a·b сдвигом и сложением (независимо от mulhi64_portable).
uint64_t ref_mulhi(uint64_t a, uint64_t b) {
    uint64_t hi = 0, lo = 0;
    for (int i = 0; i < 64; ++i) {
        if ((b >> i) & 1) {
            const uint64_t add_lo = i == 0 ? a : a << i;
            const uint64_t add_hi = i == 0 ? 0 : a >> (64 - i);
            const uint64_t nlo = lo + add_lo;
            hi += add_hi + (nlo < lo ? 1 : 0);
            lo = nlo;
        }
    }
    return hi;
}

uint64_t to_mont64(uint64_t a, uint64_t n) { return ref_mulmod(a, pow2_64_mod(n), n); }

} // namespace

TEST_CASE("fmix64 matches the splitmix64 finalizer", "[arith]") {
    // Первый выход splitmix64 с состоянием 0: fmix64(0 + golden gamma).
    REQUIRE(fmix64(0x9e3779b97f4a7c15ull) == 0xe220a8397b1dcdafull);
    REQUIRE(fmix64(0) == 0);
    REQUIRE(contrib(123, 0) == 0);
    REQUIRE(contrib(123, 1) != contrib(124, 1));
    REQUIRE(contrib(123, 1) != contrib(123, 2));
}

TEST_CASE("mulhi64 matches shift-and-add reference", "[arith]") {
    std::mt19937_64 rng(1);
    const uint64_t edge[] = {0, 1, 2, 0xffffffffull, 0x100000000ull, ~0ull, ~0ull - 1,
                             1ull << 63};
    for (uint64_t a : edge)
        for (uint64_t b : edge) {
            REQUIRE(mulhi64(a, b) == ref_mulhi(a, b));
            REQUIRE(mulhi64_portable(a, b) == ref_mulhi(a, b));
        }
    for (int i = 0; i < 20000; ++i) {
        const uint64_t a = rng(), b = rng();
        REQUIRE(mulhi64(a, b) == ref_mulhi(a, b));
        REQUIRE(mulhi64_portable(a, b) == ref_mulhi(a, b));
    }
}

TEST_CASE("modular inverses for Montgomery", "[arith]") {
    std::mt19937_64 rng(2);
    for (int i = 0; i < 10000; ++i) {
        const uint64_t n = rng() | 1;
        REQUIRE(n * inv64(n) == 1);
        const uint32_t m = static_cast<uint32_t>(rng()) | 1;
        REQUIRE(m * inv32(m) == 1u);
    }
}

TEST_CASE("pow2_64_mod equals 2^64 mod n", "[arith]") {
    std::mt19937_64 rng(3);
    for (uint64_t n : {1ull, 2ull, 3ull, 7ull, 1ull << 62, (1ull << 62) + 1, (1ull << 63) - 1,
                       1ull << 63, ~0ull}) {
        // 2^64 mod n = (2^64 - 1) mod n + 1, приведённое по n.
        const uint64_t expect = (~0ull % n + 1) % n;
        REQUIRE(pow2_64_mod(n) == expect);
    }
    for (int i = 0; i < 10000; ++i) {
        const uint64_t n = (rng() >> (rng() % 64)) | 1;
        REQUIRE(pow2_64_mod(n) == (~0ull % n + 1) % n);
    }
}

TEST_CASE("mont64_mul is a·b·2^-64 mod n for any odd n", "[arith][montgomery]") {
    std::mt19937_64 rng(4);
    for (int i = 0; i < 5000; ++i) {
        // Модули всех размеров, включая n > 2^63 (там ломался побитовый mulmod).
        const uint64_t n = (rng() >> (rng() % 63)) | 1 | (i % 3 == 0 ? 1ull << 63 : 0);
        const uint64_t ninv = inv64(n);
        const uint64_t a = rng() % n, b = rng() % n;
        const uint64_t prod = mont64_mul(to_mont64(a, n), to_mont64(b, n), n, ninv);
        // Обратное преобразование: умножение на 1 в нормальной форме.
        REQUIRE(mont64_mul(prod, 1, n, ninv) == ref_mulmod(a, b, n));
        REQUIRE(prod < n);
    }
}

TEST_CASE("mont32_mul is a·b·2^-32 mod m for any odd m", "[arith][montgomery]") {
    std::mt19937_64 rng(5);
    for (int i = 0; i < 20000; ++i) {
        const uint32_t m = static_cast<uint32_t>(rng() >> (rng() % 32)) | 1u;
        const uint32_t minv = inv32(m);
        const uint32_t a = static_cast<uint32_t>(rng() % m), b = static_cast<uint32_t>(rng() % m);
        const uint32_t r = static_cast<uint32_t>((1ull << 32) % m);
        const uint32_t am = static_cast<uint32_t>(static_cast<uint64_t>(a) * r % m);
        const uint32_t bm = static_cast<uint32_t>(static_cast<uint64_t>(b) * r % m);
        const uint32_t prod = mont32_mul(am, bm, m, minv);
        REQUIRE(prod < m);
        REQUIRE(mont32_mul(prod, 1, m, minv) == static_cast<uint32_t>(static_cast<uint64_t>(a) * b % m));
    }
}
