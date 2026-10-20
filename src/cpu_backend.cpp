#include "backend.h"

#include <cstdint>
#include <thread>
#include <vector>

namespace monkey {

namespace {

// Воркер обрабатывает непересекающиеся диапазоны индексов [begin, +batch)
// через эталонный cpu_run_range и после каждого батча публикует снимок
// числа индексов в свой слот счётчика. В горячем цикле нет atomic-RMW на
// общую память. Checksum — сумма mod 2^64, поэтому порядок не важен.
void worker(unsigned tid, unsigned total, const Config& cfg, Control& ctrl) {
    const KernelParams& kp = cfg.kernel;
    const uint64_t batch = cfg.batch_size;
    // BBP-окно конечно: индексы идут по кругу, батч не пересекает границу.
    const uint64_t wrap = (kp.kernel == KernelId::Bbp) ? bbp_units(kp.bbp) : 0;
    uint64_t begin = static_cast<uint64_t>(tid) * batch;
    const uint64_t stride = static_cast<uint64_t>(total) * batch;
    RangeResult acc;

    while (!ctrl.stop.load(std::memory_order_relaxed)) {
        uint64_t b = begin, c = batch;
        if (wrap) {
            b = begin % wrap;
            if (c > wrap - b) c = wrap - b;
        }
        const RangeResult r = cpu_run_range(kp, b, c, nullptr);
        acc.merge(r);
        ctrl.cpu_counters.record(tid, acc.count);

        if (r.first_full != UINT64_MAX) {
            ctrl.found.store(true, std::memory_order_release);
            ctrl.stop.store(true, std::memory_order_release);
            break;
        }
        begin += stride;
    }
    ctrl.cpu_counters.record(tid, acc.count);
    ctrl.checksum.fetch_add(acc.checksum, std::memory_order_relaxed); // Σ mod 2^64
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
        pool.emplace_back(worker, i, threads, std::cref(cfg), std::ref(ctrl));
    }
    for (auto& t : pool) t.join();
}

} // namespace monkey
