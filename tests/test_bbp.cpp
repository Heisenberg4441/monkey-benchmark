// BBP fixed-point (workloads/bbp_kernel.h): точность деления, цифры π против
// встроенного reference/pi_hex.txt, независимость суммы позиции от разбиения на
// блоки и окно у 10^6, которое использует suite.
#include "catch.hpp"
#include "pi_digits.h"
#include "workloads/bbp_kernel.h"

#include <cstdint>
#include <random>
#include <thread>
#include <vector>

using namespace monkey::kernel;
using monkey::pi_hex_count;
using monkey::pi_hex_digit;
using monkey::pi_hex_word;

namespace {

// Полная сумма позиции d: блоки по K членов + хвост.
uint64_t position_sum(uint64_t d, uint32_t K) {
    uint64_t s = bbp_tail(d);
    for (uint64_t k0 = 0; k0 <= d; k0 += K) s += bbp_terms(d, k0, k0 + K);
    return s;
}

} // namespace

TEST_CASE("embedded pi digits", "[bbp][pi]") {
    REQUIRE(pi_hex_count() == 1000000);
    // 3.243F6A8885A308D3...
    REQUIRE(pi_hex_word(0) == 0x243F6A88u);
    REQUIRE(pi_hex_word(8) == 0x85A308D3u);
    // Хвост файла: встраивание не обрезало последние цифры.
    REQUIRE(pi_hex_word(pi_hex_count() - 8) == 0xFD342362u);
    REQUIRE(pi_hex_digit(pi_hex_count() - 1) == 0x2);
}

TEST_CASE("bbp_frac is floor(2^64·x/m)", "[bbp][kernel]") {
    std::mt19937_64 rng(7);
    for (int i = 0; i < 100000; ++i) {
        const uint32_t m = static_cast<uint32_t>(rng() >> (rng() % 32)) | 1u;
        const uint32_t x = static_cast<uint32_t>(rng() % m);
        const uint64_t q = bbp_frac(x, m);
        // q·m <= x·2^64 < (q+1)·m, сравнение 128-битных чисел через (hi, lo).
        const uint64_t lo = q * m, hi = mulhi64(q, m);
        REQUIRE(hi <= x);
        const uint64_t rem_hi = x - hi - (lo != 0 ? 1 : 0), rem_lo = 0 - lo; // x·2^64 - q·m
        REQUIRE((rem_hi == 0 && rem_lo < m));
    }
}

TEST_CASE("BBP digits match pi_hex.txt for positions 0..2047", "[bbp][kernel]") {
    uint32_t bad = 0, first = 0;
    for (uint64_t d = 0; d < 2048; ++d) {
        if (bbp_digits(position_sum(d, 4096)) != pi_hex_word(d)) {
            if (bad++ == 0) first = static_cast<uint32_t>(d);
        }
    }
    INFO("first mismatch at position " << first);
    REQUIRE(bad == 0);
}

TEST_CASE("position sum does not depend on the block split", "[bbp][kernel][partition]") {
    for (uint64_t d : {0ull, 1ull, 7ull, 100ull, 4095ull, 4096ull, 20000ull}) {
        const uint64_t whole = position_sum(d, 1u << 30);
        REQUIRE(position_sum(d, 1) == whole);
        REQUIRE(position_sum(d, 7) == whole);
        REQUIRE(position_sum(d, 4096) == whole);
    }
}

TEST_CASE("bbp_unit covers every term of a window exactly once", "[bbp][kernel][partition]") {
    BbpParams p{};
    p.pos0 = 5000;
    p.positions = 6;
    p.block_terms = 1000;
    p.blocks = static_cast<uint32_t>((p.pos0 + p.positions - 1 + p.block_terms) / p.block_terms);
    std::vector<uint64_t> acc(p.positions, 0);
    for (uint64_t u = 0; u < static_cast<uint64_t>(p.positions) * p.blocks; ++u)
        acc[bbp_unit_position(p, u)] += bbp_unit(p, u);
    for (uint32_t w = 0; w < p.positions; ++w) {
        const uint64_t d = p.pos0 + w;
        REQUIRE(acc[w] + bbp_tail(d) == position_sum(d, 1u << 30));
        REQUIRE(bbp_digits(acc[w] + bbp_tail(d)) == pi_hex_word(d));
    }
}

TEST_CASE("BBP digits near 10^6 match pi_hex.txt", "[bbp][kernel][window]") {
    // Края окна, которое берёт suite: первые и последние позиции у конца файла.
    const uint64_t last = pi_hex_count() - 8;
    std::vector<uint64_t> positions = {last - 4096, last - 4095, last - 2, last - 1};
    std::vector<uint32_t> got(positions.size());
    std::vector<std::thread> pool;
    for (size_t i = 0; i < positions.size(); ++i)
        pool.emplace_back([&, i] { got[i] = bbp_digits(position_sum(positions[i], 4096)); });
    for (auto& t : pool) t.join();
    for (size_t i = 0; i < positions.size(); ++i) {
        INFO("position " << positions[i]);
        REQUIRE(got[i] == pi_hex_word(positions[i]));
    }
}
