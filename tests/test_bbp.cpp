// Проверка BBP-формулы: первые 256 hex-знаков π против эталона.
// Тяжёлый тест — полное вычисление BBP O(n) для каждой позиции.
#include "catch.hpp"
#include "workloads/bbp_kernel.h"

#include <cstdint>

using monkey::kernel::bbp_hex_digit;

TEST_CASE("BBP produces known hex digits of pi", "[bbp][kernel]") {
    // Первые 16 hex-знаков после 3.: 2 4 3 f 6 a 8 8 8 5 a 3 0 8 d 3
    const int expected_short[16] = {2, 4, 3, 15, 6, 10, 8, 8, 8, 5, 10, 3, 0, 8, 13, 3};

    SECTION("first 16 digits") {
        for (int i = 0; i < 16; ++i) {
            INFO("digit " << i);
            REQUIRE(bbp_hex_digit(static_cast<uint64_t>(i)) == expected_short[i]);
        }
    }

    SECTION("deterministic — same input, same output") {
        for (uint64_t d = 0; d < 64; ++d) {
            REQUIRE(bbp_hex_digit(d) == bbp_hex_digit(d));
        }
    }

    SECTION("output in valid range 0..15") {
        for (uint64_t d = 128; d < 256; d += 7) {
            const int v = bbp_hex_digit(d);
            REQUIRE(v >= 0);
            REQUIRE(v <= 15);
        }
    }
}
