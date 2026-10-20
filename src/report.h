#pragma once

#include <cstdint>
#include <limits>
#include <map>
#include <ostream>
#include <string>
#include <vector>

#include "runner.h"
#include "sysinfo.h"

// Результат прогона и его представления: таблица баллов (текст) и JSON v2
// (bench_output.schema.json). Баллы: тест — 1000·t_ref/t_median, категория —
// геометрическое среднее её тестов, итог — геометрическое среднее CPU-ST,
// CPU-MT и лучшего GPU-API. Без GPU итог не считается (флаг partial).

namespace monkey {

constexpr int kSchemaVersion = 2;
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

struct DeviceRow {
    std::string api;            // "cpu" | "vulkan" | "cuda"
    std::string name;
    std::string vendor;
    std::string driver;
    uint32_t compute_units = 0; // CPU — логических потоков
    std::string u64_mode;       // vulkan
    double init_time_s = 0;
};

struct TestOutcome {
    std::string name;
    std::string category; // ключ: cpu_st | cpu_mt | gpu_vulkan | gpu_cuda | combined | custom
    std::string backend;  // cpu | vulkan | cuda | cpu+vulkan | cpu+cuda
    uint64_t work_units = 0;
    std::vector<double> runs; // время каждого прогона, с
    double time_median = 0, time_min = 0, time_max = 0, cv = 0;
    double ops_per_sec = 0;
    uint64_t checksum = 0, expected_checksum = 0;
    uint64_t hits = 0, expected_hits = 0;
    bool has_expected = false;
    bool valid = false;
    std::string invalid_reason;
    bool unstable = false;
    double t_ref = 0;
    double score = kNaN;
    double warmup_s = 0;
    uint64_t chunk = 0;
    uint64_t cpu_units = 0, gpu_units = 0;
    // stress / custom
    bool timed = false;
    double seconds = 0;
    uint64_t passes = 0;
    SeriesStats throughput;
    double throttle_ratio = kNaN; // поздние / ранние 10% времени
    bool found = false;
    uint64_t first_full = UINT64_MAX;
};

struct Report {
    std::string mode; // standard | quick | stress | custom
    int runs = 0;
    std::string timestamp;
    SystemInfo sys;
    BuildInfo build;
    std::vector<DeviceRow> devices;
    std::vector<TestOutcome> tests;
    // Баллы (NaN — нет): по категориям, CPU, лучший GPU, итог.
    std::map<std::string, double> category_scores;
    double cpu_score = kNaN, gpu_score = kNaN, overall = kNaN;
    std::string gpu_best; // ключ категории лучшего GPU
    bool custom = false;
    bool invalid = false;  // хоть один тест недействителен
    bool unstable = false; // хоть один тест нестабилен
    bool partial = false;  // нет GPU или не все категории
};

double geomean(const std::vector<double>& v); // NaN, если пусто или есть NaN/<=0

void compute_scores(Report& r);
std::string report_json(const Report& r);
void print_scores(const Report& r, std::ostream& os);
std::string utc_timestamp();

} // namespace monkey
