#pragma once

#include <atomic>
#include <cstdint>
#include <string>
#include <vector>

#include "counters.h"
#include "gpu_device.h"
#include "kernels.h"

namespace monkey {

enum class Mode { Random, Brute };
enum class Backend { Cpu, Gpu, All };
enum class WorkloadType { Monkey, Bbp, MillerRabin };

// Полностью разобранная задача. Сравнение свёрнуто к индексам символов в
// алфавите: target_idx[i] — позиция i-го символа эталона в alphabet. Это
// убирает строки с горячего пути и одинаково работает на CPU и GPU.
struct Config {
    std::vector<std::string> alphabet;   // уникальные символы (для вывода)
    std::vector<int> target_idx;         // эталон в виде индексов алфавита
    int n = 0;                           // мощность алфавита
    int len = 0;                         // длина эталона в символах
    Mode mode = Mode::Random;
    Backend backend = Backend::Cpu;
    unsigned threads = 0;                // CPU-потоки (0 => hardware_concurrency)
    double duration = 0.0;               // лимит в секундах (0 => до совпадения)
    uint32_t seed = 0x9e3779b9u;         // seed counter-based PRNG (CLI --seed)
    uint64_t batch_size = 8192;          // гранулярность синка счётчика (CLI --batch-size)
    WorkloadType workload = WorkloadType::Monkey; // что считаем (CLI --workload)
    KernelParams kernel;                 // что именно исполняют бэкенды (kernels.h)
};

// Общее состояние, разделяемое между бэкендами и репортером.
struct Control {
    std::atomic<bool> found{false};
    std::atomic<bool> stop{false};
    std::atomic<bool> failed{false};     // ошибка бэкенда: прогон недействителен
    AttemptCounters cpu_counters;        // per-thread снимки (без contention)
    std::atomic<unsigned long long> gpu_attempts{0};
    std::atomic<unsigned long long> checksum{0}; // агрегат результата (антидот DCE)
};

// Разбиение UTF-8 строки на отдельные символы.
std::vector<std::string> split_utf8(const std::string& str);

// CPU-бэкенд: запускает cfg.threads воркеров, блокируется до ctrl.stop.
void run_cpu(const Config& cfg, Control& ctrl);

// GPU-прогон поверх GpuDevice (Vulkan или CUDA), блокируется до ctrl.stop.
// Ошибка устройства выставляет ctrl.failed и ctrl.stop.
void run_gpu(GpuDevice& dev, const Config& cfg, Control& ctrl);

} // namespace monkey
