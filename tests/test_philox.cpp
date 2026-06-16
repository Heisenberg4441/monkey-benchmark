// Известные тест-векторы (KAT) Philox4x32-10 из эталонной реализации
// D.E. Shaw Research (Random123, kat_vectors.txt). Если эти проверки зелёные —
// наш philox.h воспроизводит опубликованный алгоритм бит-в-бит.
#include "catch.hpp"
#include "philox.h"

using prng::philox4x32_10;
using prng::u32x2;
using prng::u32x4;

namespace {

void require_kat(u32x4 ctr, u32x2 key, uint32_t e0, uint32_t e1, uint32_t e2, uint32_t e3) {
    const u32x4 out = philox4x32_10(ctr, key);
    REQUIRE(out.v[0] == e0);
    REQUIRE(out.v[1] == e1);
    REQUIRE(out.v[2] == e2);
    REQUIRE(out.v[3] == e3);
}

} // namespace

TEST_CASE("Philox4x32-10 known-answer vectors (Random123)", "[philox][kat]") {
    SECTION("all-zero counter and key") {
        require_kat({{0, 0, 0, 0}}, {{0, 0}}, 0x6627e8d5u, 0xe169c58du, 0xbc57ac4cu,
                    0x9b00dbd8u);
    }
    SECTION("all-ones counter and key") {
        require_kat({{0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu}},
                    {{0xffffffffu, 0xffffffffu}}, 0x408f276du, 0x41c83b0eu, 0xa20bc7c6u,
                    0x6d5451fdu);
    }
    SECTION("digits of pi/e") {
        require_kat({{0x243f6a88u, 0x85a308d3u, 0x13198a2eu, 0x03707344u}},
                    {{0xa4093822u, 0x299f31d0u}}, 0xd16cfe09u, 0x94fdccebu, 0x5001e420u,
                    0x24126ea1u);
    }
}

TEST_CASE("philox4x32_10 is usable in constant expressions", "[philox][constexpr]") {
    // Доказывает constexpr-совместимость (требование: один и тот же код на
    // CPU compile-time и в CUDA __device__).
    constexpr u32x4 out = philox4x32_10({{0, 0, 0, 0}}, {{0, 0}});
    static_assert(out.v[0] == 0x6627e8d5u, "constexpr KAT mismatch");
    REQUIRE(out.v[0] == 0x6627e8d5u);
}

TEST_CASE("index_stream mapping is stable", "[philox][mapping]") {
    // Маппинг (seed,key,pos)->uint32 завязан на расположение слов counter/key.
    // Эта проверка ловит случайное изменение свёртки, которое разъехало бы
    // CPU и GPU (у них общий index_stream).
    REQUIRE(prng::index_stream(0u, 0u, 0u) == philox4x32_10({{0, 0, 0, 0}}, {{0, 0}}).v[0]);
    REQUIRE(prng::index_stream(0x9e3779b9u, 7u, 3u) ==
            philox4x32_10({{7u, 3u, 0u, 0u}}, {{0x9e3779b9u, 0u}}).v[0]);
}
