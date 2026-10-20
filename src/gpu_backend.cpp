#include "backend.h"

#include <chrono>
#include <cstdio>

namespace monkey {

// Прогон на GPU поверх GpuDevice: последовательные диапазоны индексов с
// размером, который удваивается, пока диспатч короче ~20 мс. Так ни один
// диспатч не длится секунды (TDR на Windows) и stop срабатывает быстро.
// Ошибка устройства останавливает весь прогон и помечает его как failed.
void run_gpu(GpuDevice& dev, const Config& cfg, Control& ctrl) {
    try {
        dev.prepare(cfg.kernel);
        const uint64_t wrap = (cfg.kernel.kernel == KernelId::Bbp) ? bbp_units(cfg.kernel.bbp) : 0;
        uint64_t begin = 0, chunk = 1u << 16, done = 0; // 2^16 потоков — заполнить любой GPU
        while (!ctrl.stop.load(std::memory_order_relaxed)) {
            uint64_t b = begin, c = chunk;
            if (wrap) {
                b = begin % wrap;
                if (c > wrap - b) c = wrap - b;
            }
            const auto t0 = std::chrono::steady_clock::now();
            const RangeResult r = dev.run_range(b, c);
            const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            done += r.count;
            ctrl.gpu_attempts.store(done, std::memory_order_relaxed);
            if (r.first_full != UINT64_MAX) {
                ctrl.found.store(true, std::memory_order_release);
                ctrl.stop.store(true, std::memory_order_release);
                break;
            }
            begin += c;
            if (dt < 0.02 && chunk < (1ull << 40)) chunk *= 2;
        }
    } catch (const GpuError& e) {
        std::fprintf(stderr, "[error] %s\n", e.what());
        ctrl.failed.store(true, std::memory_order_release);
        ctrl.stop.store(true, std::memory_order_release);
    }
}

} // namespace monkey
