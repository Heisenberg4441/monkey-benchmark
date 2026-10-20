// Эталонный CPU-исполнитель kernel'ов (kernels.h): канон маппинга индекс ->
// кандидат, семантика результата и главное свойство checksum — RangeResult не
// зависит от того, как диапазон разрезан на куски и в каком порядке они
// посчитаны (так CPU-потоки, CUDA и Vulkan могут делить одну работу).
#include "catch.hpp"
#include "kernels.h"
#include "sieve.h"
#include "workloads/monkey_kernel.h"

#include <algorithm>
#include <cstdint>
#include <random>
#include <vector>

using namespace monkey;

namespace {

// «The quick brown fox jumps over the lazy dog»: алфавит из 28 символов
// (пробел, 'T' и все 26 строчных), индексы как их строит main —
// позиции в отсортированном списке уникальных символов.
KernelParams fox(KernelId k) {
    const std::string s = "The quick brown fox jumps over the lazy dog";
    std::vector<char> alpha(s.begin(), s.end());
    std::sort(alpha.begin(), alpha.end());
    alpha.erase(std::unique(alpha.begin(), alpha.end()), alpha.end());
    KernelParams p;
    p.kernel = k;
    p.n = static_cast<uint32_t>(alpha.size());
    for (char c : s)
        p.target.push_back(static_cast<int32_t>(std::find(alpha.begin(), alpha.end(), c) - alpha.begin()));
    return p;
}

// Результат того же диапазона, посчитанный случайными кусками в случайном порядке.
RangeResult shuffled(const KernelParams& p, uint64_t begin, uint64_t count, uint64_t seed,
                     std::vector<uint64_t>* acc) {
    std::mt19937_64 rng(seed);
    std::vector<std::pair<uint64_t, uint64_t>> pieces;
    for (uint64_t b = begin; b < begin + count;) {
        const uint64_t c = std::min<uint64_t>(1 + rng() % 997, begin + count - b);
        pieces.emplace_back(b, c);
        b += c;
    }
    std::shuffle(pieces.begin(), pieces.end(), rng);
    RangeResult total;
    for (auto [b, c] : pieces) total.merge(cpu_run_range(p, b, c, acc ? acc->data() : nullptr));
    return total;
}

} // namespace

TEST_CASE("fox reference has n = 28, len = 43", "[kernels]") {
    const KernelParams p = fox(KernelId::MonkeyRandom);
    REQUIRE(p.n == 28);
    REQUIRE(p.len() == 43);
}

TEST_CASE("canonical candidate mapping is frozen", "[kernels][monkey][mapping]") {
    // Канон: Philox counter = {pos, i_lo, i_hi, 0}, key = {seed, 0}, word0 mod n.
    // Изменение этих значений меняет все кандидаты и все golden suite.
    const uint32_t expect[3][8] = {{26, 2, 13, 25, 26, 7, 1, 24},
                                   {8, 17, 12, 6, 14, 0, 1, 5},
                                   {11, 13, 0, 3, 17, 10, 10, 14}};
    for (uint64_t i = 0; i < 3; ++i)
        for (uint32_t pos = 0; pos < 8; ++pos)
            REQUIRE(kernel::monkey_char(0x9e3779b9u, i, pos, 27) == expect[i][pos]);
    // Старшее слово индекса участвует в маппинге.
    REQUIRE(kernel::monkey_char(0x9e3779b9u, 1ull << 32, 0, 1u << 31) !=
            kernel::monkey_char(0x9e3779b9u, 0, 0, 1u << 31));
}

TEST_CASE("monkey random result is the matched prefix length", "[kernels][monkey]") {
    KernelParams p = fox(KernelId::MonkeyRandom);
    const uint64_t i = 123456789;
    // Эталон = сам кандидат i -> полное совпадение; меняем последний символ -> len-1.
    for (uint32_t pos = 0; pos < p.len(); ++pos)
        p.target[pos] = static_cast<int32_t>(kernel::monkey_char(p.seed, i, pos, p.n));
    REQUIRE(kernel::monkey_random_prefix(p.seed, i, p.target.data(), p.len(), p.n) == p.len());
    p.target.back() = (p.target.back() + 1) % static_cast<int32_t>(p.n);
    REQUIRE(kernel::monkey_random_prefix(p.seed, i, p.target.data(), p.len(), p.n) == p.len() - 1);
    p.target.front() = (p.target.front() + 1) % static_cast<int32_t>(p.n);
    REQUIRE(kernel::monkey_random_prefix(p.seed, i, p.target.data(), p.len(), p.n) == 0);
}

TEST_CASE("monkey brute result is the matched suffix length", "[kernels][monkey]") {
    KernelParams p;
    p.kernel = KernelId::MonkeyBrute;
    p.n = 3;
    p.target = {1, 0, 2, 1, 2};
    const uint64_t code = (((1 * 3 + 0) * 3 + 2) * 3 + 1) * 3 + 2; // 104
    REQUIRE(kernel::monkey_brute_suffix(code, p.target.data(), 5, 3) == 5);
    REQUIRE(kernel::monkey_brute_suffix(code + 1, p.target.data(), 5, 3) == 0);
    REQUIRE(kernel::monkey_brute_suffix(code - 3, p.target.data(), 5, 3) == 1);
    // Поиск: первое полное совпадение — ровно код эталона.
    const RangeResult r = cpu_run_range(p, 0, 1000, nullptr);
    REQUIRE(r.first_full == code);
    REQUIRE(r.count == 1000);
    // fox: n^len > 2^64, но любой 64-битный индекс — корректный кандидат.
    const KernelParams f = fox(KernelId::MonkeyBrute);
    REQUIRE(cpu_run_range(f, 1ull << 63, 100000, nullptr).count == 100000);
}

TEST_CASE("RangeResult does not depend on the partition", "[kernels][partition]") {
    struct Case {
        KernelParams p;
        uint64_t begin, count;
    };
    KernelParams bbp;
    bbp.kernel = KernelId::Bbp;
    bbp.bbp.pos0 = 20000;
    bbp.bbp.positions = 5;
    bbp.bbp.block_terms = 512;
    bbp.bbp.blocks = (20000 + 4 + 512) / 512;
    KernelParams mr;
    mr.kernel = KernelId::MillerRabin;
    mr.mr_base = (1ull << 62) + 1;
    mr.mr_step = 2;
    const Case cases[] = {
        {fox(KernelId::MonkeyRandom), 1ull << 40, 300000},
        {fox(KernelId::MonkeyBrute), 0, 300000},
        {bbp, 0, bbp_units(bbp.bbp)},
        {mr, 1000, 50000},
    };
    for (const Case& c : cases) {
        INFO("kernel " << kernel_name(c.p.kernel));
        std::vector<uint64_t> acc1(c.p.bbp.positions + 1, 0), acc2(c.p.bbp.positions + 1, 0);
        const RangeResult whole = cpu_run_range(c.p, c.begin, c.count, acc1.data());
        REQUIRE(whole.count == c.count);
        REQUIRE(whole.hits > 0);
        for (uint64_t seed : {1u, 2u, 3u}) {
            std::fill(acc2.begin(), acc2.end(), 0);
            REQUIRE(shuffled(c.p, c.begin, c.count, seed, &acc2) == whole);
            REQUIRE(acc2 == acc1);
        }
    }
}

TEST_CASE("miller-rabin kernel counts primes of its window", "[kernels][millerrabin]") {
    KernelParams mr;
    mr.kernel = KernelId::MillerRabin;
    mr.mr_base = (1ull << 40) + 1;
    mr.mr_step = 2;
    const uint64_t count = 50000;
    const auto sieve = test_sieve::segment(mr.mr_base, mr.mr_base + 2 * count);
    uint64_t primes = 0;
    for (uint64_t i = 0; i < count; ++i) primes += sieve[2 * i];
    REQUIRE(cpu_run_range(mr, 0, count, nullptr).hits == primes);
}

TEST_CASE("kernel self-check", "[kernels]") {
    std::string why;
    REQUIRE(kernel_selfcheck(fox(KernelId::MonkeyRandom), &why));
    REQUIRE(kernel_selfcheck(fox(KernelId::MonkeyBrute), &why));
    KernelParams bad = fox(KernelId::MonkeyRandom);
    bad.target[0] = 99;
    REQUIRE_FALSE(kernel_selfcheck(bad, &why));
    KernelParams bbp;
    bbp.kernel = KernelId::Bbp;
    bbp.bbp = bbp_window(1024);
    REQUIRE(kernel_selfcheck(bbp, &why));
    KernelParams mr;
    mr.kernel = KernelId::MillerRabin;
    REQUIRE(kernel_selfcheck(mr, &why));
}
