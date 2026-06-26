#include "backend.h"
#include "workload.h"

#include <cstdint>
#include <thread>
#include <vector>

namespace monkey {

namespace {

// Воркер ведёт локальный WorkloadResult (без аллокаций между батчами) и после
// каждого батча публикует снимок iterations в свой слот счётчика. В горячем
// цикле нет ни одного atomic-RMW на общую память. Потоки берут непересекающиеся
// диапазоны counter'ов; execute_batch — const, поэтому один экземпляр workload'а
// безопасно делится между всеми воркерами.
void worker(unsigned tid, unsigned total, const Config& cfg, Control& ctrl,
            const IWorkload& wl) {
    const uint64_t batch = cfg.batch_size;
    const uint64_t stride = static_cast<uint64_t>(total) * batch;
    uint64_t counter = static_cast<uint64_t>(tid) * batch;
    WorkloadResult res;

    while (!ctrl.stop.load(std::memory_order_relaxed)) {
        wl.execute_batch(counter, batch, res);
        ctrl.cpu_counters.record(tid, res.iterations);

        if (res.solution_found) {
            ctrl.found.store(true, std::memory_order_release);
            ctrl.stop.store(true, std::memory_order_release);
            break;
        }
        counter += stride;
    }
    ctrl.cpu_counters.record(tid, res.iterations);
    ctrl.checksum.fetch_xor(res.checksum, std::memory_order_relaxed);
}

} // namespace

void run_cpu(const Config& cfg, Control& ctrl) {
    unsigned threads = cfg.threads;
    if (threads == 0) threads = std::thread::hardware_concurrency();
    if (threads == 0) threads = 1;

    // Слоты счётчиков по одному на воркер; публикуется до старта потоков.
    ctrl.cpu_counters.resize(threads);

    auto wl = make_workload(cfg);
    if (!wl) { // main уже валидирует workload; страховка от рассинхрона
        ctrl.stop.store(true, std::memory_order_release);
        return;
    }
    wl->prepare(cfg);

    std::vector<std::thread> pool;
    pool.reserve(threads);
    for (unsigned i = 0; i < threads; ++i) {
        pool.emplace_back(worker, i, threads, std::cref(cfg), std::ref(ctrl), std::cref(*wl));
    }
    for (auto& t : pool) t.join();
}

} // namespace monkey
