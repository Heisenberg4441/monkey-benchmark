#include "backend.h"
#include "prng.h"

#include <cstdint>
#include <thread>
#include <vector>

namespace monkey {

namespace {

// Воркер ведёт локальный счётчик в регистре/стеке и сбрасывает его снимок в
// свой (никем больше не читаемый/не пишемый) слот раз в batch_size итераций.
// В горячем цикле НЕТ ни одного atomic-RMW на общую память — снимок это
// relaxed store в собственную линию кэша. Контеншна на счётчике нет.
void random_worker(unsigned thread_id, const Config& cfg, Control& ctrl) {
    const int len = cfg.len;
    const int n = cfg.n;
    // Свой поток случайных чисел на воркер; key — номер попытки внутри потока.
    // seed детерминирован (cfg.seed, CLI --seed): прогон воспроизводим, а
    // вывод бит-в-бит совпадает с CUDA/Vulkan при том же thread_seed.
    const uint32_t thread_seed = cfg.seed + thread_id * 2654435761u;
    const uint64_t batch = cfg.batch_size;
    uint32_t key = 0;
    uint64_t local = 0;
    uint64_t since_flush = 0;

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
        ++local;

        if (++since_flush >= batch) {
            ctrl.cpu_counters.record(thread_id, local);
            since_flush = 0;
        }

        if (match) {
            ctrl.cpu_counters.record(thread_id, local);
            ctrl.found.store(true, std::memory_order_release);
            ctrl.stop.store(true, std::memory_order_release);
            return;
        }
    }
    ctrl.cpu_counters.record(thread_id, local);
}

void brute_worker(unsigned thread_id, unsigned total_threads, const Config& cfg, Control& ctrl) {
    const int n = cfg.n;
    const int len = cfg.len;
    const uint64_t batch = cfg.batch_size;

    const unsigned long long chunk = 1000000ULL;
    unsigned long long start_idx = static_cast<unsigned long long>(thread_id) * chunk;
    const unsigned long long stride = static_cast<unsigned long long>(total_threads) * chunk;

    std::vector<int> comb(len, 0);
    uint64_t local = 0;
    uint64_t since_flush = 0;

    while (!ctrl.stop.load(std::memory_order_relaxed)) {
        // Стартовое число чанка -> комбинация в системе счисления с основанием n.
        unsigned long long tmp = start_idx;
        for (int i = 0; i < len; ++i) {
            comb[len - 1 - i] = static_cast<int>(tmp % n);
            tmp /= n;
        }

        for (unsigned long long i = 0; i < chunk; ++i) {
            if (ctrl.stop.load(std::memory_order_relaxed)) {
                ctrl.cpu_counters.record(thread_id, local);
                return;
            }

            bool match = true;
            for (int j = 0; j < len; ++j) {
                if (comb[j] != cfg.target_idx[j]) {
                    match = false;
                    break;
                }
            }

            ++local;
            if (++since_flush >= batch) {
                ctrl.cpu_counters.record(thread_id, local);
                since_flush = 0;
            }

            if (match) {
                ctrl.cpu_counters.record(thread_id, local);
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
    ctrl.cpu_counters.record(thread_id, local);
}

} // namespace

void run_cpu(const Config& cfg, Control& ctrl) {
    unsigned threads = cfg.threads;
    if (threads == 0) threads = std::thread::hardware_concurrency();
    if (threads == 0) threads = 1;

    // Слоты счётчиков по одному на воркер; публикуется до старта потоков.
    ctrl.cpu_counters.resize(threads);

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
