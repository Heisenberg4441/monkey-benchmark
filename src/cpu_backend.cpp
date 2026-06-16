#include "backend.h"
#include "prng.h"

#include <cstdint>
#include <thread>
#include <vector>

namespace monkey {

namespace {

// Размер локального батча перед атомарным обновлением глобального счётчика.
// Снижает contention на общем atomic в горячем цикле.
constexpr unsigned long long kBatch = 8192;

void random_worker(unsigned thread_id, const Config& cfg, Control& ctrl) {
    const int len = cfg.len;
    const int n = cfg.n;
    // Свой поток случайных чисел на воркер; key — номер попытки внутри потока.
    // seed детерминирован (cfg.seed, CLI --seed): прогон воспроизводим, а
    // вывод бит-в-бит совпадает с CUDA/Vulkan при том же thread_seed.
    const uint32_t thread_seed = cfg.seed + thread_id * 2654435761u;
    uint32_t key = 0;
    unsigned long long local = 0;

    while (!ctrl.stop.load(std::memory_order_relaxed)) {
        bool match = true;
        for (int i = 0; i < len; ++i) {
            int c = rand_index(thread_seed, key, static_cast<uint32_t>(i), n);
            if (c != cfg.target_idx[i]) {
                match = false;
                break;
            }
        }
        ++key;

        if (++local >= kBatch) {
            ctrl.cpu_attempts.fetch_add(local, std::memory_order_relaxed);
            local = 0;
        }

        if (match) {
            ctrl.cpu_attempts.fetch_add(local, std::memory_order_relaxed);
            ctrl.found.store(true, std::memory_order_release);
            ctrl.stop.store(true, std::memory_order_release);
            return;
        }
    }
    ctrl.cpu_attempts.fetch_add(local, std::memory_order_relaxed);
}

void brute_worker(unsigned thread_id, unsigned total_threads, const Config& cfg, Control& ctrl) {
    const int n = cfg.n;
    const int len = cfg.len;

    const unsigned long long chunk = 1000000ULL;
    unsigned long long start_idx = static_cast<unsigned long long>(thread_id) * chunk;
    const unsigned long long stride = static_cast<unsigned long long>(total_threads) * chunk;

    std::vector<int> comb(len, 0);
    unsigned long long local = 0;

    while (!ctrl.stop.load(std::memory_order_relaxed)) {
        // Стартовое число чанка -> комбинация в системе счисления с основанием n.
        unsigned long long tmp = start_idx;
        for (int i = 0; i < len; ++i) {
            comb[len - 1 - i] = static_cast<int>(tmp % n);
            tmp /= n;
        }

        for (unsigned long long i = 0; i < chunk; ++i) {
            if (ctrl.stop.load(std::memory_order_relaxed)) {
                ctrl.cpu_attempts.fetch_add(local, std::memory_order_relaxed);
                return;
            }

            bool match = true;
            for (int j = 0; j < len; ++j) {
                if (comb[j] != cfg.target_idx[j]) {
                    match = false;
                    break;
                }
            }

            if (++local >= kBatch) {
                ctrl.cpu_attempts.fetch_add(local, std::memory_order_relaxed);
                local = 0;
            }

            if (match) {
                ctrl.cpu_attempts.fetch_add(local, std::memory_order_relaxed);
                ctrl.found.store(true, std::memory_order_release);
                ctrl.stop.store(true, std::memory_order_release);
                return;
            }

            // Инкремент n-ричного числа "в столбик".
            for (int j = len; j-- > 0;) {
                if (++comb[j] < n) break;
                comb[j] = 0;
            }
        }
        start_idx += stride;
    }
    ctrl.cpu_attempts.fetch_add(local, std::memory_order_relaxed);
}

} // namespace

void run_cpu(const Config& cfg, Control& ctrl) {
    unsigned threads = cfg.threads;
    if (threads == 0) threads = std::thread::hardware_concurrency();
    if (threads == 0) threads = 1;

    std::vector<std::thread> pool;
    pool.reserve(threads);
    for (unsigned i = 0; i < threads; ++i) {
        if (cfg.mode == Mode::Random) {
            pool.emplace_back(random_worker, i, std::cref(cfg), std::ref(ctrl));
        } else {
            pool.emplace_back(brute_worker, i, threads, std::cref(cfg), std::ref(ctrl));
        }
    }
    for (auto& t : pool) t.join();
}

} // namespace monkey
