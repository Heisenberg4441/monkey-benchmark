#pragma once

#include <atomic>
#include <cstdint>
#include <string>
#include <vector>

namespace monkey {

enum class Mode { Random, Brute };
enum class Backend { Cpu, Gpu, All };

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
};

// Общее состояние, разделяемое между бэкендами и репортером.
struct Control {
    std::atomic<bool> found{false};
    std::atomic<bool> stop{false};
    std::atomic<unsigned long long> cpu_attempts{0};
    std::atomic<unsigned long long> gpu_attempts{0};
};

// Разбиение UTF-8 строки на отдельные символы.
std::vector<std::string> split_utf8(const std::string& str);

// CPU-бэкенд: запускает cfg.threads воркеров, блокируется до ctrl.stop.
void run_cpu(const Config& cfg, Control& ctrl);

// CUDA-бэкенд. Реализация в cuda_backend.cu (USE_CUDA) либо заглушка в
// cuda_stub.cpp. cuda_available() сообщает, доступен ли GPU в рантайме.
bool cuda_available();
void run_cuda(const Config& cfg, Control& ctrl);

// Vulkan-бэкенд (любой GPU: AMD/Intel/Apple/NVIDIA). Реализация в
// vulkan_backend.cpp (USE_VULKAN) либо заглушка в vulkan_stub.cpp.
bool vulkan_available();
void run_vulkan(const Config& cfg, Control& ctrl);

} // namespace monkey
