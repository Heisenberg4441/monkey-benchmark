// Проверяет корректность безатомного (для горячего цикла) счётчика попыток:
// сумма снимков по всем потокам должна точно равняться числу выполненных
// итераций. Это аналог acceptance-теста Фазы 2 «финальная сумма счётчиков ==
// число итераций», но детерминированный (фиксированное число итераций вместо
// настенного времени — никакой флакости от таймера).
#include "catch.hpp"
#include "counters.h"

#include <atomic>
#include <cstdint>
#include <thread>
#include <vector>

using monkey::AttemptCounters;

TEST_CASE("AttemptCounters: single thread sum equals iterations", "[counters]") {
    AttemptCounters c;
    c.resize(1);
    constexpr uint64_t kIters = 1'000'000;
    uint64_t local = 0;
    for (uint64_t i = 0; i < kIters; ++i) {
        ++local;
        if (i % 8192 == 0) c.record(0, local); // периодический флаш, как в воркере
    }
    c.record(0, local); // финальный флаш
    REQUIRE(c.total() == kIters);
}

TEST_CASE("AttemptCounters: concurrent sum is lossless", "[counters]") {
    constexpr unsigned kThreads = 8;
    constexpr uint64_t kIters = 2'000'000;
    constexpr uint64_t kBatch = 8192;

    AttemptCounters c;
    c.resize(kThreads);

    // Барьер старта, чтобы потоки реально работали параллельно (стресс на
    // отсутствие false sharing / потери обновлений).
    std::atomic<unsigned> ready{0};
    std::atomic<bool> go{false};

    auto worker = [&](unsigned tid) {
        ready.fetch_add(1, std::memory_order_acq_rel);
        while (!go.load(std::memory_order_acquire)) {
        }
        uint64_t local = 0;
        uint64_t since = 0;
        for (uint64_t i = 0; i < kIters; ++i) {
            ++local;
            if (++since >= kBatch) {
                c.record(tid, local);
                since = 0;
            }
        }
        c.record(tid, local);
    };

    std::vector<std::thread> pool;
    pool.reserve(kThreads);
    for (unsigned t = 0; t < kThreads; ++t) pool.emplace_back(worker, t);
    while (ready.load(std::memory_order_acquire) < kThreads) {
    }
    go.store(true, std::memory_order_release);
    for (auto& th : pool) th.join();

    REQUIRE(c.total() == static_cast<uint64_t>(kThreads) * kIters);
}

TEST_CASE("AttemptCounters: mid-run snapshot is monotonic and bounded", "[counters]") {
    // Снимок во время работы не превышает истинное число итераций и не убывает —
    // репортёр читает корректную (хоть и слегка отстающую) сумму.
    AttemptCounters c;
    c.resize(1);
    uint64_t local = 0;
    uint64_t prev_snapshot = 0;
    for (uint64_t i = 0; i < 100'000; ++i) {
        ++local;
        if (i % 1000 == 0) {
            c.record(0, local);
            const uint64_t snap = c.total();
            REQUIRE(snap >= prev_snapshot);
            REQUIRE(snap <= local);
            prev_snapshot = snap;
        }
    }
}
