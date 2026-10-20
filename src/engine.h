#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "counters.h"
#include "gpu_device.h"
#include "kernels.h"

// Движок исполнения: одна сессия на устройство и общий диспетчер чанков.
//
// - Воркеры (CPU-потоки и хост-поток GPU) создаются один раз и живут всю
//   сессию: прогрев и замеры идут подряд, без пересоздания устройства.
// - Работа — диапазон индексов [begin, end); воркеры берут из него чанки
//   атомарным fetch_add (один атомик на чанк, не на операцию). Диспетчер общий
//   для всех устройств, поэтому в -all нет дубликатов, а балансировка
//   получается сама.
// - Старт — по общему барьеру; время — steady_clock от отпускания барьера до
//   завершения последнего чанка. Инициализация устройств в замер не входит.
// - Размер чанка подстраивается на прогреве (adaptive): ~15 мс на CPU-чанк,
//   ~30 мс на GPU-диспатч. Стоп по времени или по совпадению срабатывает не
//   позже чем через один чанк.
// - Ошибка любого воркера останавливает всех; run() возвращает её текстом.

namespace monkey {

// Завершённый чанк: время начала и конца относительно старта задания.
struct ChunkSample {
    double t_begin;
    double t_end;
    uint64_t units;
    uint32_t worker; // 0..cpu-1 — CPU-потоки, cpu — GPU
};

struct JobSpec {
    const KernelParams* kp = nullptr;
    uint64_t begin = 0;
    uint64_t end = 0;              // индексы [begin, end)
    double max_seconds = 0;        // > 0 — остановиться по времени (прогрев, timed)
    bool stop_on_full_match = false;
    bool adaptive = false;         // подстройка размера чанка (прогрев)
    bool keep_samples = false;
    bool use_cpu = true;
    unsigned cpu_workers = 0;      // 0 — все потоки пула
    bool use_gpu = false;
};

struct JobResult {
    bool completed = false; // обработан весь диапазон
    double seconds = 0;     // от старта до конца последнего чанка
    RangeResult res;
    std::vector<uint64_t> bbp_acc; // суммы BBP по позициям (CPU + GPU)
    std::vector<ChunkSample> samples;
    uint64_t cpu_units = 0;
    uint64_t gpu_units = 0;
    std::string error;      // непусто — бэкенд упал, результат недействителен
};

// Текущие размеры чанков в индексах (для CPU — на поток, для GPU — на диспатч).
struct ChunkPlan {
    uint64_t cpu = 0;
    uint64_t gpu = 0;
};

class Engine {
public:
    // cpu_threads — размер CPU-пула (0 — без CPU), gpu — устройство или nullptr.
    Engine(unsigned cpu_threads, GpuDevice* gpu);
    ~Engine();
    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;

    unsigned cpu_threads() const { return cpu_threads_; }
    GpuDevice* gpu() const { return gpu_; }

    // Цели подстройки чанка, секунды.
    double cpu_target_s = 0.015;
    double gpu_target_s = 0.030;
    ChunkPlan plan; // 0 — начальная оценка по kernel'у

    // Синхронное задание. progress(units, seconds) вызывается из этого потока
    // примерно раз в 250 мс, пока задание идёт.
    JobResult run(const JobSpec& spec,
                  const std::function<void(uint64_t, double)>& progress = {});

private:
    struct Job;
    void worker_main(unsigned id);
    void work(unsigned id, Job& job);

    unsigned cpu_threads_;
    GpuDevice* gpu_;
    std::vector<std::thread> threads_;
    std::mutex m_;
    std::condition_variable cv_job_, cv_done_;
    uint64_t generation_ = 0;
    bool quit_ = false;
    unsigned pending_ = 0;
    Job* job_ = nullptr;
};

// Начальная оценка размера чанка для kernel'а (до подстройки).
uint64_t initial_chunk(KernelId k, bool gpu);

} // namespace monkey
