#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "engine.h"
#include "kernels.h"

// Режимы прогона поверх Engine.
//   fixed-work — фиксированный диапазон [0, N): прогрев с подстройкой чанка, затем
//                R повторов в той же сессии; балл считается из времени.
//   timed      — работа до истечения времени (стресс, custom): проходы по
//                фиксированному окну, сэмплы — по завершённым чанкам.

namespace monkey {

// Какие устройства участвуют.
struct DeviceSel {
    bool use_cpu = true;
    unsigned cpu_workers = 0; // 0 — весь пул
    bool use_gpu = false;
};

struct SeriesStats {
    size_t n = 0;
    double min = 0, median = 0, mean = 0, p95 = 0, p99 = 0, max = 0, stddev = 0, cv = 0;
};
SeriesStats series_stats(std::vector<double> v);

// Пропускная способность по корзинам bin_s секунд: единицы каждого чанка
// равномерно распределяются по его интервалу [t_begin, t_end]. Неполная
// последняя корзина отбрасывается.
std::vector<double> throughput_series(const std::vector<ChunkSample>& samples, double bin_s,
                                      double total_s);

struct FixedRun {
    double warmup_s = 0;               // фактическая длительность прогрева
    std::vector<double> times;         // время каждого повтора, с
    std::vector<RangeResult> results;  // результат каждого повтора
    std::vector<std::vector<uint64_t>> bbp_acc; // суммы BBP каждого повтора
    uint64_t cpu_units = 0, gpu_units = 0;      // последнего повтора
    ChunkPlan plan;
    std::string error;
};

using Progress = std::function<void(const char* phase, uint64_t done, uint64_t total, double t)>;

FixedRun run_fixed(Engine& e, const KernelParams& kp, uint64_t N, const DeviceSel& sel,
                   double warmup_s, int repeats, const Progress& progress = {});

struct TimedRun {
    double seconds = 0;
    RangeResult res;      // по всем проходам
    uint64_t passes = 0;  // завершённых проходов окна
    uint64_t cpu_units = 0, gpu_units = 0;
    bool found = false;   // полное совпадение (monkey)
    std::vector<ChunkSample> samples; // время — от начала замера
    // Результаты завершённых проходов окна — каждый сверяется с golden.
    std::vector<RangeResult> pass_results;
    std::vector<std::vector<uint64_t>> pass_bbp;
    std::string error;
};

// range_end — конец окна: проходы [0, range_end) повторяются, пока не выйдет
// время (seconds > 0). seconds == 0 — один проход (для поиска «до совпадения»
// range_end может быть UINT64_MAX). tune == false — чанк берётся из e.plan как
// есть (custom --batch-size), без прогрева и подстройки.
TimedRun run_timed(Engine& e, const KernelParams& kp, uint64_t range_end, double seconds,
                   bool stop_on_match, const DeviceSel& sel, double warmup_s,
                   const Progress& progress = {}, bool tune = true);

} // namespace monkey
