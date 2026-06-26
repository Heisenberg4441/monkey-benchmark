#pragma once

#include <atomic>
#include <cstdint>
#include <memory>

namespace monkey {

// Per-thread счётчики попыток без contention. Каждый воркер пишет ТОЛЬКО в свой
// слот (relaxed store) и только при флаше — раз в batch_size итераций, не в самом
// горячем шаге. Репортёр (главный поток) суммирует снимки (relaxed load).
// Слоты выровнены по линии кэша, чтобы исключить false sharing между ядрами.
//
// resize() вызывается один раз перед стартом воркеров; count_ публикуется с
// release, total() читает его с acquire — это делает видимым массив slots_ для
// репортёра, который стартует конкурентно.
class AttemptCounters {
public:
    void resize(unsigned n) {
        slots_ = std::make_unique<Slot[]>(n);
        count_.store(n, std::memory_order_release);
    }

    // Снимок текущего локального счётчика воркера tid.
    void record(unsigned tid, uint64_t value) noexcept {
        slots_[tid].v.store(value, std::memory_order_relaxed);
    }

    uint64_t total() const noexcept {
        const unsigned n = count_.load(std::memory_order_acquire);
        uint64_t sum = 0;
        for (unsigned i = 0; i < n; ++i) {
            sum += slots_[i].v.load(std::memory_order_relaxed);
        }
        return sum;
    }

    unsigned size() const noexcept { return count_.load(std::memory_order_acquire); }

private:
    // Каждый слот живёт на своей линии кэша (64 байта), чтобы исключить false sharing.
    // Явный padding вместо alignas(64) — обходим MSVC-баг с over-aligned new[].
    struct Slot {
        std::atomic<uint64_t> v{0};
        char pad[56];
    };
    static_assert(sizeof(Slot) == 64, "Slot must be exactly one cache line");

    std::unique_ptr<Slot[]> slots_;
    std::atomic<unsigned> count_{0};
};

} // namespace monkey
