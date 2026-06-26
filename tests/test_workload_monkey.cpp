// MonkeyWorkload под интерфейсом IWorkload: verify()-gate, точный счёт
// итераций и корректный ранний выход на найденном решении.
#include "catch.hpp"
#include "backend.h"
#include "workload.h"
#include "workload_monkey.h"
#include "workloads/monkey_kernel.h"

#include <cstdint>
#include <vector>

using monkey::Config;
using monkey::Mode;
using monkey::MonkeyWorkload;
using monkey::WorkloadResult;

namespace {

Config make_cfg(Mode mode, std::vector<int> target, int n) {
    Config c;
    c.mode = mode;
    c.n = n;
    c.len = static_cast<int>(target.size());
    c.target_idx = std::move(target);
    c.seed = 0x9e3779b9u;
    return c;
}

// Кодирует эталон в counter для brute (как это делает worker'ный перебор).
uint64_t encode_brute(const std::vector<int>& target, int n) {
    uint64_t c = 0;
    for (int d : target) c = c * static_cast<uint64_t>(n) + static_cast<uint64_t>(d);
    return c;
}

} // namespace

TEST_CASE("MonkeyWorkload verify passes for brute and random", "[workload][monkey]") {
    MonkeyWorkload w;
    SECTION("brute") {
        Config c = make_cfg(Mode::Brute, {1, 0, 2, 1}, 3);
        w.prepare(c);
        REQUIRE(w.verify());
    }
    SECTION("random") {
        Config c = make_cfg(Mode::Random, {1, 0, 2, 1, 2, 0}, 3);
        w.prepare(c);
        REQUIRE(w.verify());
    }
}

TEST_CASE("MonkeyWorkload brute finds the exact encoding counter", "[workload][monkey]") {
    const std::vector<int> target = {1, 0, 2, 1, 2}; // n=3
    Config c = make_cfg(Mode::Brute, target, 3);
    MonkeyWorkload w;
    w.prepare(c);

    const uint64_t expected = encode_brute(target, 3);
    WorkloadResult res;
    // Один батч, накрывающий искомый counter.
    w.execute_batch(0, expected + 10, res);

    REQUIRE(res.solution_found);
    REQUIRE(res.solution_counter == expected);
    // Перебор останавливается на найденном: обработано ровно expected+1 counter'ов.
    REQUIRE(res.iterations == expected + 1);
}

// Эталон из 24 символов при n=3: пространство 3^24 ≈ 2.8e11, поэтому совпадение
// в батче 1e5 статистически невозможно (ожидание ~3e-7).
const std::vector<int> kLongTarget = {0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2,
                                      0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2};

TEST_CASE("MonkeyWorkload counts every iteration when no match", "[workload][monkey]") {
    Config c = make_cfg(Mode::Random, kLongTarget, 3);
    MonkeyWorkload w;
    w.prepare(c);

    WorkloadResult res;
    constexpr uint64_t kBatch = 100000;
    w.execute_batch(0, kBatch, res);

    REQUIRE_FALSE(res.solution_found);
    REQUIRE(res.iterations == kBatch);
}

TEST_CASE("MonkeyWorkload accumulates across batches", "[workload][monkey]") {
    Config c = make_cfg(Mode::Random, kLongTarget, 3);
    MonkeyWorkload w;
    w.prepare(c);

    WorkloadResult res;
    w.execute_batch(0, 1000, res);
    w.execute_batch(1000, 1000, res);
    REQUIRE(res.iterations == 2000);
}

TEST_CASE("monkey_char is in range and deterministic", "[workload][monkey][kernel]") {
    for (uint64_t counter = 0; counter < 1000; counter += 37) {
        for (uint32_t pos = 0; pos < 8; ++pos) {
            const int a = monkey::kernel::monkey_char(123u, counter, pos, 26);
            const int b = monkey::kernel::monkey_char(123u, counter, pos, 26);
            REQUIRE(a == b);
            REQUIRE(a >= 0);
            REQUIRE(a < 26);
        }
    }
}
