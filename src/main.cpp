#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <io.h>
#include <windows.h>
#define isatty _isatty
#define fileno _fileno
#else
#include <unistd.h>
#endif

#include "engine.h"
#include "gpu_device.h"
#include "kernels.h"
#include "report.h"
#include "runner.h"
#include "suite_v1.h"
#include "sysinfo.h"

// CLI бенчмарка. Подкоманды: run (suite v1 и баллы), list, info, verify, custom.
// Старый синтаксис `<reference_file> -cpu -d 10` — это custom. Вывод CLI на
// английском: бенчмарк международный.

namespace monkey {

void reference_to_target(const std::string& text, uint32_t* n, std::vector<int32_t>* target);

namespace {

using namespace suite_v1;

enum class GpuApi { Auto, Cuda, Vulkan };

[[noreturn]] void usage_error(const std::string& msg) {
    std::cerr << "error: " << msg << "\nSee `monkey_bench --help`.\n";
    std::exit(1);
}

void print_help() {
    std::cout <<
        "monkey_bench — Infinite Monkey Benchmark, suite v" << kVersion << "\n\n"
        "Usage:\n"
        "  monkey_bench [run] [options]          run the suite and print scores\n"
        "  monkey_bench list                     list the suite tests\n"
        "  monkey_bench info                     system, build and compute devices\n"
        "  monkey_bench verify [options]         check every kernel on every backend\n"
        "  monkey_bench custom [file] [options]  custom workload: ops/s only, not comparable\n"
        "  monkey_bench <file> [options]         legacy syntax, same as custom\n\n"
        "run options:\n"
        "  --standard | --quick | --stress   3 runs per test (default) | 1 run | timed stress\n"
        "  --cpu | --gpu | --all             CPU categories | GPU categories | both + CPU+GPU\n"
        "  --gpu-api auto|cuda|vulkan        GPU APIs to test (default: every available one)\n"
        "  --test NAME                       run only this test (repeatable)\n"
        "  -d, --duration SEC                stress: seconds per test and category (default 30)\n"
        "  --json FILE                       write the JSON v2 report (\"-\" for stdout)\n"
        "  --vk-allow-cpu                    allow software Vulkan devices (lavapipe)\n\n"
        "custom options:\n"
        "  -w, --workload monkey|bbp|miller-rabin    -m, --mode random|brute (monkey)\n"
        "  -cpu | -gpu | -all, -b, --backend cpu|gpu|all\n"
        "  -t, --threads N    -d, --duration SEC    --warmup SEC (default 2)    --seed N\n"
        "  --batch-size N     fixed chunk size in indices (default: tuned)\n"
        "  --output-format text|json    --json FILE    --gpu-api ...    --vk-allow-cpu\n\n"
        "Without -d a monkey search runs until the exact match is found.\n";
}

bool is_tty(FILE* f) { return isatty(fileno(f)) != 0; }

uint64_t parse_u64(const std::string& key, const std::string& v) {
    char* end = nullptr;
    errno = 0;
    const unsigned long long x = std::strtoull(v.c_str(), &end, 0);
    if (v.empty() || *end != '\0' || errno != 0 || v[0] == '-') usage_error("invalid value for " + key + ": " + v);
    return x;
}

double parse_seconds(const std::string& key, const std::string& v) {
    char* end = nullptr;
    const double x = std::strtod(v.c_str(), &end);
    if (v.empty() || *end != '\0' || !(x >= 0) || !std::isfinite(x)) usage_error("invalid value for " + key + ": " + v);
    return x;
}

// Значение опции: "--key=value" или следующий аргумент.
std::string take(const std::vector<std::string>& a, size_t& i, const std::string& key) {
    const auto eq = a[i].find('=');
    if (eq != std::string::npos) return a[i].substr(eq + 1);
    if (i + 1 >= a.size()) usage_error("missing value for " + key);
    return a[++i];
}

std::string key_of(const std::string& arg) { return arg.substr(0, arg.find('=')); }

GpuApi parse_api(const std::string& v) {
    if (v == "auto") return GpuApi::Auto;
    if (v == "cuda") return GpuApi::Cuda;
    if (v == "vulkan") return GpuApi::Vulkan;
    usage_error("unknown GPU API: " + v);
}

std::string fmt_rate(double ops) {
    char buf[32];
    if (ops >= 1e9) std::snprintf(buf, sizeof buf, "%.2f G/s", ops / 1e9);
    else if (ops >= 1e6) std::snprintf(buf, sizeof buf, "%.2f M/s", ops / 1e6);
    else std::snprintf(buf, sizeof buf, "%.0f /s", ops);
    return buf;
}

// --- устройства ----------------------------------------------------------------

struct GpuSlot {
    std::string key; // gpu_cuda | gpu_vulkan
    std::unique_ptr<GpuDevice> dev;
};

// Открывает GPU выбранных API. errors — причины, по которым API недоступен.
std::vector<GpuSlot> open_gpus(GpuApi api, bool allow_cpu, std::vector<std::string>* errors) {
    std::vector<GpuSlot> out;
    std::string err;
    if (api == GpuApi::Auto || api == GpuApi::Cuda) {
        if (auto d = create_cuda_device(CudaOptions{}, &err)) out.push_back({"gpu_cuda", std::move(d)});
        else errors->push_back(err);
    }
    if (api == GpuApi::Auto || api == GpuApi::Vulkan) {
        VulkanOptions vo;
        vo.allow_cpu_device = allow_cpu;
        if (auto d = create_vulkan_device(vo, &err)) out.push_back({"gpu_vulkan", std::move(d)});
        else errors->push_back(err);
    }
    return out;
}

DeviceRow cpu_row(const SystemInfo& si, unsigned threads) {
    DeviceRow d;
    d.api = "cpu";
    d.name = si.cpu_model;
    d.compute_units = threads;
    return d;
}

DeviceRow gpu_row(const GpuDevice& g) {
    const GpuInfo& i = g.info();
    DeviceRow d;
    d.api = i.api;
    d.name = i.name;
    d.vendor = i.vendor;
    d.driver = i.driver;
    d.compute_units = i.compute_units;
    d.u64_mode = i.u64_mode;
    d.init_time_s = i.init_time_s;
    return d;
}

std::string describe_cpu(const SystemInfo& si) {
    std::ostringstream os;
    os << (si.cpu_model.empty() ? "unknown CPU" : si.cpu_model) << " · " << si.cores_logical << " threads";
    if (si.cores_performance && si.cores_efficiency)
        os << " (" << si.cores_performance << "P + " << si.cores_efficiency << "E)";
    if (si.ram_bytes) os << " · " << std::fixed << std::setprecision(1) << si.ram_bytes / 1073741824.0 << " GiB";
    os << " · " << si.os << " " << si.os_version << " " << si.arch;
    return os.str();
}

std::string describe_gpu(const GpuDevice& g) {
    const GpuInfo& i = g.info();
    std::ostringstream os;
    os << i.name << " · " << (i.api == "cuda" ? "CUDA" : "Vulkan") << " (" << i.driver;
    if (!i.u64_mode.empty()) os << ", u64 " << i.u64_mode;
    os << ")" << (i.cpu_device ? " · software device" : "") << " · init " << std::fixed << std::setprecision(2)
       << i.init_time_s << " s";
    return os.str();
}

// Строка прогресса в stderr (только в терминале).
Progress make_progress(bool enabled, const std::string& label, uint64_t total) {
    if (!enabled) return {};
    return [label, total](const char* phase, uint64_t done, uint64_t, double t) {
        std::fprintf(stderr, "\r  %-34s %-6s %5.1f s", label.c_str(), phase, t);
        if (total && std::string(phase) == "run")
            std::fprintf(stderr, "  %5.1f%%", 100.0 * static_cast<double>(done) / static_cast<double>(total));
        std::fprintf(stderr, "        ");
        std::fflush(stderr);
    };
}

void clear_progress(bool enabled) {
    if (enabled) std::fprintf(stderr, "\r%80s\r", "");
}

// --- проверка результата --------------------------------------------------------

// Сверка одного прогона теста с golden. false + reason, если не совпало.
bool check_run(const TestDef& t, Category c, const KernelParams& kp, uint64_t N, const RangeResult& r,
               const std::vector<uint64_t>& bbp_acc, std::string* reason) {
    if (r.count != N) {
        *reason = "processed " + std::to_string(r.count) + " of " + std::to_string(N) + " units";
        return false;
    }
    const Golden& g = t.golden[c];
    if (g.hits == 0) {
        *reason = "no golden value in this build";
        return false;
    }
    if (r.checksum != g.checksum || r.hits != g.hits) {
        *reason = "checksum mismatch";
        return false;
    }
    if (t.kernel == KernelId::Bbp && bbp_verify(kp.bbp, bbp_acc).bad_positions != 0) {
        *reason = "pi digits mismatch";
        return false;
    }
    return true;
}

void fill_times(TestOutcome& o, const std::vector<double>& times) {
    o.runs = times;
    const SeriesStats s = series_stats(times);
    o.time_median = s.median;
    o.time_min = s.min;
    o.time_max = s.max;
    o.cv = s.cv;
    o.unstable = times.size() > 1 && s.cv > kUnstableCv;
}

// --- run ------------------------------------------------------------------------

struct CategoryPlan {
    std::string key;     // cpu_st | cpu_mt | gpu_cuda | gpu_vulkan | combined
    Category cat;        // размеры и golden
    Engine* engine;
    DeviceSel sel;
    std::string backend;
};

TestOutcome run_fixed_test(const TestDef& t, const CategoryPlan& cp, int runs, double warmup, bool tty) {
    TestOutcome o;
    o.name = t.name;
    o.category = cp.key;
    o.backend = cp.backend;
    const KernelParams kp = make_params(t, cp.cat);
    const uint64_t N = work_units(t, cp.cat);
    o.work_units = N;
    const std::string label = std::string("[") + cp.key + "] " + t.name;
    const FixedRun fr = run_fixed(*cp.engine, kp, N, cp.sel, warmup, runs, make_progress(tty, label, N));
    clear_progress(tty);
    if (!fr.error.empty()) {
        std::cerr << "error: " << label << ": " << fr.error << "\nThe run is aborted; no result.\n";
        std::exit(1);
    }
    fill_times(o, fr.times);
    o.warmup_s = fr.warmup_s;
    o.chunk = cp.sel.use_cpu ? fr.plan.cpu : fr.plan.gpu;
    o.cpu_units = fr.cpu_units;
    o.gpu_units = fr.gpu_units;
    o.ops_per_sec = o.time_median > 0 ? static_cast<double>(N) / o.time_median : 0;
    o.checksum = fr.results[0].checksum;
    o.hits = fr.results[0].hits;
    o.expected_checksum = t.golden[cp.cat].checksum;
    o.expected_hits = t.golden[cp.cat].hits;
    o.has_expected = t.golden[cp.cat].hits != 0;
    o.valid = true;
    for (size_t i = 0; i < fr.results.size() && o.valid; ++i)
        o.valid = check_run(t, cp.cat, kp, N, fr.results[i], fr.bbp_acc[i], &o.invalid_reason);
    o.t_ref = cp.key == "combined" ? 0 : t.t_ref[cp.cat];
    if (o.valid && o.t_ref > 0) o.score = 1000.0 * o.t_ref / o.time_median;
    return o;
}

TestOutcome run_stress_test(const TestDef& t, const CategoryPlan& cp, double seconds, bool tty) {
    TestOutcome o;
    o.name = t.name;
    o.category = cp.key;
    o.backend = cp.backend;
    o.timed = true;
    const KernelParams kp = make_params(t, cp.cat);
    const uint64_t N = work_units(t, cp.cat);
    o.work_units = N;
    const std::string label = std::string("[") + cp.key + "] " + t.name;
    const TimedRun tr = run_timed(*cp.engine, kp, N, seconds, false, cp.sel, kWarmupQuick,
                                  make_progress(tty, label, 0));
    clear_progress(tty);
    if (!tr.error.empty()) {
        std::cerr << "error: " << label << ": " << tr.error << "\nThe run is aborted; no result.\n";
        std::exit(1);
    }
    o.seconds = tr.seconds;
    o.passes = tr.passes;
    o.cpu_units = tr.cpu_units;
    o.gpu_units = tr.gpu_units;
    o.ops_per_sec = tr.seconds > 0 ? static_cast<double>(tr.res.count) / tr.seconds : 0;
    o.checksum = tr.res.checksum;
    o.hits = tr.res.hits;
    // Каждый завершённый проход окна — полноценная проверка.
    o.valid = true;
    for (size_t i = 0; i < tr.pass_results.size() && o.valid; ++i)
        o.valid = check_run(t, cp.cat, kp, N, tr.pass_results[i], tr.pass_bbp[i], &o.invalid_reason);
    const double bin = std::max(0.25, seconds / 120.0);
    const std::vector<double> series = throughput_series(tr.samples, bin, tr.seconds);
    o.throughput = series_stats(series);
    if (series.size() >= 10) {
        const size_t k = series.size() / 10;
        double early = 0, late = 0;
        for (size_t i = 0; i < k; ++i) {
            early += series[i];
            late += series[series.size() - 1 - i];
        }
        o.throttle_ratio = early > 0 ? late / early : kNaN;
    }
    return o;
}

int cmd_run(const std::vector<std::string>& args) {
    std::string mode = "standard";
    bool want_cpu = true, want_gpu = true, want_combined = false, gpu_explicit = false;
    GpuApi api = GpuApi::Auto;
    bool allow_cpu_vk = false;
    double duration = 30;
    std::string json_path;
    std::vector<std::string> only;
    for (size_t i = 0; i < args.size(); ++i) {
        const std::string& a = args[i];
        const std::string k = key_of(a);
        if (a == "--standard") mode = "standard";
        else if (a == "--quick") mode = "quick";
        else if (a == "--stress") mode = "stress";
        else if (a == "--cpu") { want_cpu = true; want_gpu = false; }
        else if (a == "--gpu") { want_cpu = false; want_gpu = true; gpu_explicit = true; }
        else if (a == "--all") { want_cpu = want_gpu = want_combined = true; }
        else if (k == "--gpu-api") { api = parse_api(take(args, i, k)); gpu_explicit = gpu_explicit || api != GpuApi::Auto; }
        else if (k == "--test") only.push_back(take(args, i, k));
        else if (k == "-d" || k == "--duration") duration = parse_seconds(k, take(args, i, k));
        else if (k == "--json") json_path = take(args, i, k);
        else if (a == "--vk-allow-cpu") allow_cpu_vk = true;
        else usage_error("unknown option for run: " + a);
    }
    for (const auto& name : only) {
        bool known = false;
        for (const TestDef& t : kTests) known |= name == t.name;
        if (!known) usage_error("unknown test: " + name + " (see `monkey_bench list`)");
    }
    const bool json_stdout = json_path == "-";
    std::ostream& text = json_stdout ? std::cerr : std::cout;
    const bool tty = is_tty(stderr);

    Report rep;
    rep.mode = mode;
    rep.runs = mode == "standard" ? kRunsStandard : (mode == "quick" ? kRunsQuick : 0);
    rep.timestamp = utc_timestamp();
    rep.sys = query_system();
    rep.build = query_build();
    const unsigned threads = std::max(1u, rep.sys.cores_logical ? rep.sys.cores_logical : std::thread::hardware_concurrency());

    std::vector<GpuSlot> gpus;
    if (want_gpu) {
        std::vector<std::string> errors;
        gpus = open_gpus(api, allow_cpu_vk, &errors);
        if (gpus.empty()) {
            if (gpu_explicit) {
                std::cerr << "error: no GPU available";
                for (const auto& e : errors) std::cerr << "\n  " << e;
                std::cerr << "\n";
                return 1;
            }
            text << "note: no GPU available, GPU categories skipped";
            for (const auto& e : errors) text << "\n  " << e;
            text << "\n";
        }
    }

    const BuildInfo& b = rep.build;
    text << "monkey_bench " << b.version << " (" << b.git_sha << ") · suite v" << kVersion << " · " << mode
         << "\nCPU:  " << describe_cpu(rep.sys) << "\n";
    for (const auto& g : gpus) text << "GPU:  " << describe_gpu(*g.dev) << "\n";
    if (b.debug) text << "warning: debug build — scores are not comparable\n";
    if (b.native_arch) text << "warning: -march=native build — scores are not comparable\n";
    text << "\n";

    rep.devices.push_back(cpu_row(rep.sys, threads));
    for (const auto& g : gpus) rep.devices.push_back(gpu_row(*g.dev));

    std::unique_ptr<Engine> cpu_engine;
    if (want_cpu) cpu_engine = std::make_unique<Engine>(threads, nullptr);
    std::vector<std::unique_ptr<Engine>> gpu_engines;
    for (auto& g : gpus) gpu_engines.push_back(std::make_unique<Engine>(0, g.dev.get()));
    std::unique_ptr<Engine> combined;
    // CPU+GPU: одно ядро остаётся хост-потоку GPU, иначе 14 CPU-воркеров на 14
    // ядрах задерживают подачу диспатчей и GPU простаивает (замер в BENCH.md).
    if (want_combined && !gpus.empty())
        combined = std::make_unique<Engine>(threads > 1 ? threads - 1 : 1, gpus.front().dev.get());

    std::vector<CategoryPlan> plans;
    if (cpu_engine) {
        plans.push_back({"cpu_st", kCpuSt, cpu_engine.get(), {true, 1, false}, "cpu"});
        plans.push_back({"cpu_mt", kCpuMt, cpu_engine.get(), {true, 0, false}, "cpu"});
    }
    for (size_t i = 0; i < gpus.size(); ++i)
        plans.push_back({gpus[i].key, kGpu, gpu_engines[i].get(), {false, 0, true}, gpus[i].dev->info().api});
    if (combined)
        plans.push_back({"combined", kGpu, combined.get(), {true, 0, true}, "cpu+" + gpus.front().dev->info().api});

    const double warmup = mode == "standard" ? kWarmupStandard : kWarmupQuick;
    for (const CategoryPlan& cp : plans) {
        for (const TestDef& t : kTests) {
            if (!only.empty() && std::find(only.begin(), only.end(), t.name) == only.end()) continue;
            TestOutcome o = mode == "stress" ? run_stress_test(t, cp, duration, tty)
                                             : run_fixed_test(t, cp, rep.runs, warmup, tty);
            char line[256];
            if (o.timed) {
                std::snprintf(line, sizeof line, "  %-11s %-14s %6.1f s  %3llu passes  %12s  min/max %.3f  throttle %.3f  %s",
                              cp.key.c_str(), t.name, o.seconds, static_cast<unsigned long long>(o.passes),
                              fmt_rate(o.ops_per_sec).c_str(),
                              o.throughput.mean > 0 ? o.throughput.min / o.throughput.max : 0.0, o.throttle_ratio,
                              o.valid ? "valid" : ("INVALID: " + o.invalid_reason).c_str());
            } else {
                std::snprintf(line, sizeof line, "  %-11s %-14s %7.3f s  CV %4.1f%%  %12s  %s", cp.key.c_str(),
                              t.name, o.time_median, o.runs.size() > 1 ? o.cv * 100 : 0.0,
                              fmt_rate(o.ops_per_sec).c_str(),
                              o.valid ? (o.unstable ? "valid, unstable" : "valid")
                                      : ("INVALID: " + o.invalid_reason).c_str());
            }
            text << line << "\n" << std::flush;
            rep.tests.push_back(std::move(o));
        }
    }

    if (mode != "stress") {
        compute_scores(rep);
        print_scores(rep, text);
    } else {
        for (const auto& t : rep.tests) rep.invalid |= !t.valid;
        rep.partial = true;
    }
    if (!json_path.empty()) {
        const std::string js = report_json(rep);
        if (json_stdout) {
            std::cout << js;
        } else {
            std::ofstream f(json_path, std::ios::binary);
            f << js;
            if (!f) {
                std::cerr << "error: cannot write " << json_path << "\n";
                return 1;
            }
            text << "\nJSON report: " << json_path << "\n";
        }
    }
    return rep.invalid ? 2 : 0;
}

// --- list / info ----------------------------------------------------------------

int cmd_list() {
    std::cout << "Suite v" << kVersion << " — fixed work per test and category, score = 1000 · t_ref / t\n"
              << "Reference machine (1000 points): " << kReferenceMachine << "\n\n";
    for (const TestDef& t : kTests) {
        std::cout << t.name << " — " << t.about << "\n";
        for (int c = 0; c < kCategories; ++c) {
            const auto cat = static_cast<Category>(c);
            std::cout << "  " << std::left << std::setw(18) << category_title(cat) << std::right << std::setw(16)
                      << work_units(t, cat) << " units";
            if (t.kernel == KernelId::Bbp) std::cout << " (" << t.size[c] << " positions)";
            std::cout << "   t_ref " << std::fixed << std::setprecision(3) << t.t_ref[c] << " s\n";
        }
    }
    return 0;
}

int cmd_info(const std::vector<std::string>& args) {
    bool allow_cpu_vk = false;
    for (const auto& a : args) {
        if (a == "--vk-allow-cpu") allow_cpu_vk = true;
        else usage_error("unknown option for info: " + a);
    }
    const SystemInfo si = query_system();
    const BuildInfo b = query_build();
    std::cout << "monkey_bench " << b.version << " (" << b.git_sha << "), suite v" << kVersion << "\n"
              << "Build:   " << b.compiler << ", " << b.build_type << ", arith "
              << (b.portable_arith ? "portable" : "native") << (b.native_arch ? ", -march=native" : "")
              << "\n         flags: " << b.flags << "\n"
              << "System:  " << si.os << " " << si.os_version << " (kernel " << si.kernel << "), " << si.arch << "\n"
              << "CPU:     " << describe_cpu(si) << ", " << si.cores_physical << " physical cores\n";
    std::cout << "Vulkan:  " << (vulkan_compiled() ? "built in" : "not built in");
    const auto vk_devices = list_vulkan_devices();
    if (!vulkan_runtime().empty()) std::cout << " (runtime " << vulkan_runtime() << ")";
    for (const auto& d : vk_devices) std::cout << "\n         " << d;
    std::cout << "\nCUDA:    " << (cuda_compiled() ? "built in" : "not built in");
    for (const auto& d : list_cuda_devices()) std::cout << "\n         " << d;
    std::cout << "\n";
    std::vector<std::string> errors;
    const auto gpus = open_gpus(GpuApi::Auto, allow_cpu_vk, &errors);
    for (const auto& g : gpus) std::cout << "Selected " << describe_gpu(*g.dev) << "\n";
    for (const auto& e : errors) std::cout << "         " << e << "\n";
    return 0;
}

// --- verify ---------------------------------------------------------------------

int cmd_verify(const std::vector<std::string>& args) {
    GpuApi api = GpuApi::Auto;
    bool allow_cpu_vk = false;
    for (size_t i = 0; i < args.size(); ++i) {
        const std::string k = key_of(args[i]);
        if (k == "--gpu-api") api = parse_api(take(args, i, k));
        else if (args[i] == "--vk-allow-cpu") allow_cpu_vk = true;
        else usage_error("unknown option for verify: " + args[i]);
    }
    // Бэкенды: CPU, каждый GPU-API; Vulkan — в обоих вариантах 64-битной арифметики.
    struct Backend {
        std::string name;
        std::unique_ptr<GpuDevice> dev; // nullptr — CPU
    };
    std::vector<Backend> backends;
    backends.push_back({"cpu", nullptr});
    std::string err;
    std::vector<std::string> skipped;
    if (api == GpuApi::Auto || api == GpuApi::Cuda) {
        if (auto d = create_cuda_device(CudaOptions{}, &err)) backends.push_back({"cuda " + d->info().name, std::move(d)});
        else skipped.push_back(err);
    }
    if (api == GpuApi::Auto || api == GpuApi::Vulkan) {
        for (bool emu : {false, true}) {
            VulkanOptions vo;
            vo.allow_cpu_device = allow_cpu_vk;
            vo.force_emulated_u64 = emu;
            auto d = create_vulkan_device(vo, &err);
            if (!d) {
                skipped.push_back(err);
                break;
            }
            if (!emu && d->info().u64_mode != "native") continue;
            backends.push_back({"vulkan " + d->info().name + " (u64 " + d->info().u64_mode + ")", std::move(d)});
        }
    }

    const unsigned threads = std::max(1u, std::thread::hardware_concurrency());
    Engine cpu(threads, nullptr);
    int failures = 0;
    std::cout << "Verifying suite v" << kVersion << " kernels against golden values\n";
    for (Backend& be : backends) {
        std::cout << "\n" << be.name << "\n";
        for (const VerifyCase& vc : kVerify) {
            const TestDef& t = kTests[vc.test];
            const KernelParams kp = make_params(t, vc.category);
            RangeResult r;
            std::string error;
            try {
                if (be.dev) {
                    be.dev->prepare(kp);
                    for (uint64_t b = vc.begin; b < vc.begin + vc.count;) {
                        const uint64_t c = std::min<uint64_t>(vc.begin + vc.count - b, 1u << 20);
                        r.merge(be.dev->run_range(b, c));
                        b += c;
                    }
                } else {
                    JobSpec s;
                    s.kp = &kp;
                    s.begin = vc.begin;
                    s.end = vc.begin + vc.count;
                    const JobResult jr = cpu.run(s);
                    error = jr.error;
                    r = jr.res;
                }
            } catch (const std::exception& e) {
                error = e.what();
            }
            const bool ok = error.empty() && r.count == vc.count && r.checksum == vc.golden.checksum &&
                            r.hits == vc.golden.hits && vc.golden.hits != 0;
            failures += !ok;
            std::cout << "  " << (ok ? "PASS" : "FAIL") << "  " << std::left << std::setw(14) << t.name
                      << " [" << vc.begin << ", +" << vc.count << ")";
            if (!ok) {
                char buf[96];
                std::snprintf(buf, sizeof buf, "  got 0x%016llx/%llu, want 0x%016llx/%llu",
                              static_cast<unsigned long long>(r.checksum), static_cast<unsigned long long>(r.hits),
                              static_cast<unsigned long long>(vc.golden.checksum),
                              static_cast<unsigned long long>(vc.golden.hits));
                std::cout << (error.empty() ? buf : "  " + error);
            }
            std::cout << "\n";
        }
    }
    // Цифры π: окно CPU-ST целиком на CPU против встроенного pi_hex.txt.
    {
        const TestDef& t = kTests[2];
        const KernelParams kp = make_params(t, kCpuSt);
        JobSpec s;
        s.kp = &kp;
        s.end = work_units(t, kCpuSt);
        const JobResult jr = cpu.run(s);
        const BbpVerdict v = bbp_verify(kp.bbp, jr.bbp_acc);
        const bool ok = jr.error.empty() && v.bad_positions == 0;
        failures += !ok;
        std::cout << "\ncpu\n  " << (ok ? "PASS" : "FAIL") << "  bbp-pi digits: " << kp.bbp.positions
                  << " positions from " << kp.bbp.pos0 << " vs embedded pi_hex.txt";
        if (!ok) std::cout << " (" << v.bad_positions << " wrong)";
        std::cout << "\n";
    }
    for (const auto& s : skipped) std::cout << "\nskipped: " << s << "\n";
    std::cout << "\n" << (failures ? "FAILED: " + std::to_string(failures) + " check(s)" : "All checks passed") << "\n";
    return failures ? 2 : 0;
}

// --- custom ---------------------------------------------------------------------

int cmd_custom(const std::vector<std::string>& args) {
    std::string file, workload = "monkey", mode = "random", backend = "cpu", json_path;
    GpuApi api = GpuApi::Auto;
    bool allow_cpu_vk = false, json_stdout = false;
    unsigned threads = 0;
    double duration = 0, warmup = 2.0;
    uint32_t seed = kSeed;
    uint64_t batch = 0;
    for (size_t i = 0; i < args.size(); ++i) {
        const std::string& a = args[i];
        const std::string k = key_of(a);
        if (a == "-cpu" || a == "-gpu" || a == "-all") backend = a.substr(1);
        else if (k == "-b" || k == "--backend") {
            backend = take(args, i, k);
            if (backend != "cpu" && backend != "gpu" && backend != "all") usage_error("unknown backend: " + backend);
        } else if (k == "-w" || k == "--workload") {
            workload = take(args, i, k);
            if (workload != "monkey" && workload != "bbp" && workload != "miller-rabin")
                usage_error("unknown workload: " + workload);
        } else if (k == "-m" || k == "--mode") {
            mode = take(args, i, k);
            if (mode != "random" && mode != "brute") usage_error("unknown mode: " + mode);
        } else if (k == "--gpu-api") api = parse_api(take(args, i, k));
        else if (k == "-t" || k == "--threads") threads = static_cast<unsigned>(parse_u64(k, take(args, i, k)));
        else if (k == "-d" || k == "--duration") duration = parse_seconds(k, take(args, i, k));
        else if (k == "--warmup") warmup = parse_seconds(k, take(args, i, k));
        else if (k == "--seed") seed = static_cast<uint32_t>(parse_u64(k, take(args, i, k)));
        else if (k == "--batch-size") {
            batch = parse_u64(k, take(args, i, k));
            if (batch == 0) usage_error("--batch-size must be > 0");
        } else if (k == "--output-format") {
            const std::string v = take(args, i, k);
            if (v == "json") json_stdout = true;
            else if (v != "text") usage_error("unknown output format: " + v);
        } else if (k == "--json") json_path = take(args, i, k);
        else if (a == "--vk-allow-cpu") allow_cpu_vk = true;
        else if (!a.empty() && a[0] == '-') usage_error("unknown option: " + a);
        else if (file.empty()) file = a;
        else usage_error("unexpected argument: " + a);
    }
    if (json_path == "-") json_stdout = true;
    std::ostream& text = json_stdout ? std::cerr : std::cout;

    KernelParams kp;
    std::string reference;
    uint64_t range_end = UINT64_MAX;
    std::string test_name = "custom-" + workload;
    if (workload == "monkey") {
        if (file.empty()) usage_error("monkey workload needs a reference file");
        std::ifstream f(file, std::ios::binary);
        if (!f) usage_error("cannot open " + file);
        reference.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
        while (!reference.empty() && (reference.back() == '\n' || reference.back() == '\r')) reference.pop_back();
        if (reference.empty()) usage_error(file + " is empty");
        kp.kernel = mode == "brute" ? KernelId::MonkeyBrute : KernelId::MonkeyRandom;
        kp.seed = seed;
        reference_to_target(reference, &kp.n, &kp.target);
        if (kp.n < 2) usage_error("the reference needs at least 2 distinct characters");
        test_name += "-" + mode;
        if (kp.kernel == KernelId::MonkeyBrute) {
            // Пространство n^len: если влезает в 64 бита — это и есть конец перебора.
            uint64_t space = 1;
            bool fits = true;
            for (uint32_t i = 0; i < kp.len() && fits; ++i) {
                if (space > UINT64_MAX / kp.n) fits = false;
                else space *= kp.n;
            }
            if (fits) range_end = space;
            else if (duration <= 0)
                usage_error("brute: n^len exceeds 2^64, the match is unreachable; use -d or random mode");
            else text << "note: brute: n^len exceeds 2^64, the match is unreachable — throughput only\n";
        }
    } else {
        // bbp / miller-rabin — параметры suite (категория CPU-MT), без баллов.
        const TestDef& t = kTests[workload == "bbp" ? 2 : 3];
        kp = make_params(t, kCpuMt);
        range_end = work_units(t, kCpuMt);
        if (duration <= 0) duration = 10;
    }
    std::string why;
    if (!kernel_selfcheck(kp, &why)) {
        std::cerr << "error: kernel self-check failed (" << why << ")\n";
        return 1;
    }

    const SystemInfo si = query_system();
    const unsigned pool = threads ? threads : std::max(1u, si.cores_logical);
    const bool use_gpu = backend != "cpu", use_cpu = backend != "gpu";
    std::unique_ptr<GpuDevice> gpu;
    if (use_gpu) {
        std::vector<std::string> errors;
        auto gpus = open_gpus(api, allow_cpu_vk, &errors);
        if (gpus.empty()) {
            std::cerr << "error: no GPU available";
            for (const auto& e : errors) std::cerr << "\n  " << e;
            std::cerr << "\n";
            return 1;
        }
        gpu = std::move(gpus.front().dev); // auto: CUDA, если есть, иначе Vulkan
    }
    Engine engine(use_cpu ? pool : 0, gpu.get());
    if (batch) {
        engine.plan.cpu = engine.plan.gpu = batch;
        warmup = 0; // фиксированный чанк — без подстройки
    }
    const std::string be = use_cpu && use_gpu ? "cpu+" + gpu->info().api : (use_gpu ? gpu->info().api : "cpu");
    if (!reference.empty())
        text << "Reference: " << (reference.size() > 60 ? reference.substr(0, 60) + "…" : reference) << " ("
             << kp.len() << " chars, alphabet " << kp.n << ")\n";
    text << "Workload:  " << test_name << " · backend " << be;
    if (use_cpu) text << " · " << pool << " CPU threads";
    text << "\n";
    if (gpu) text << "GPU:       " << describe_gpu(*gpu) << "\n";
    char dur[32];
    std::snprintf(dur, sizeof dur, "%.1f s", duration);
    text << "Duration:  " << (duration > 0 ? std::string(dur) : std::string("until match")) << " · warmup "
         << warmup << " s\n" << std::flush;

    const bool tty = is_tty(stderr);
    Progress prog;
    if (tty) {
        prog = [](const char*, uint64_t done, uint64_t, double t) {
            std::fprintf(stderr, "\r  %.1f s · %llu ops · %s          ", t, static_cast<unsigned long long>(done),
                         t > 0 ? fmt_rate(static_cast<double>(done) / t).c_str() : "-");
            std::fflush(stderr);
        };
    }
    const DeviceSel sel{use_cpu, 0, use_gpu};
    const bool search = kp.kernel == KernelId::MonkeyRandom || kp.kernel == KernelId::MonkeyBrute;
    // --batch-size: фиксированный чанк из plan, без прогрева и подстройки.
    const TimedRun tr = run_timed(engine, kp, range_end, duration, search, sel, warmup, prog, batch == 0);
    clear_progress(tty);
    if (!tr.error.empty()) {
        std::cerr << "error: " << tr.error << "\nThe run is aborted; no result.\n";
        return 1;
    }

    TestOutcome o;
    o.name = test_name;
    o.category = "custom";
    o.backend = be;
    o.timed = true;
    o.seconds = tr.seconds;
    o.passes = tr.passes;
    o.work_units = tr.res.count;
    o.cpu_units = tr.cpu_units;
    o.gpu_units = tr.gpu_units;
    o.ops_per_sec = tr.seconds > 0 ? static_cast<double>(tr.res.count) / tr.seconds : 0;
    o.checksum = tr.res.checksum;
    o.hits = tr.res.hits;
    o.found = tr.found;
    o.first_full = tr.res.first_full;
    o.valid = true;
    o.chunk = use_cpu ? engine.plan.cpu : engine.plan.gpu;
    const double bin = std::max(0.1, tr.seconds / 100.0);
    o.throughput = series_stats(throughput_series(tr.samples, bin, tr.seconds));

    if (o.found)
        text << "MATCH FOUND at index " << o.first_full << " after " << tr.seconds << " s\n";
    else if (kp.kernel == KernelId::MonkeyRandom || kp.kernel == KernelId::MonkeyBrute)
        text << "No exact match within " << tr.seconds << " s\n";
    text << "Operations: " << tr.res.count << " in " << tr.seconds << " s · average " << fmt_rate(o.ops_per_sec)
         << "\n";
    if (o.throughput.n > 1)
        text << "Throughput per " << bin << " s (from completed chunks): min " << fmt_rate(o.throughput.min)
             << ", median " << fmt_rate(o.throughput.median) << ", max " << fmt_rate(o.throughput.max) << ", CV "
             << std::setprecision(2) << o.throughput.cv * 100 << "%\n";
    text << "Custom run: ops/s only, not comparable with suite scores.\n";

    if (json_stdout || !json_path.empty()) {
        Report rep;
        rep.mode = "custom";
        rep.custom = true;
        rep.timestamp = utc_timestamp();
        rep.sys = si;
        rep.build = query_build();
        if (use_cpu) rep.devices.push_back(cpu_row(si, pool));
        if (gpu) rep.devices.push_back(gpu_row(*gpu));
        rep.tests.push_back(o);
        rep.partial = true;
        const std::string js = report_json(rep);
        if (json_stdout) std::cout << js;
        if (!json_path.empty() && json_path != "-") {
            std::ofstream f(json_path, std::ios::binary);
            f << js;
        }
    }
    return 0;
}

} // namespace

int cli(int argc, char** argv) {
    std::vector<std::string> args(argv + 1, argv + argc);
    if (!args.empty() && (args[0] == "-h" || args[0] == "--help" || args[0] == "help")) {
        print_help();
        return 0;
    }
    if (!args.empty() && args[0] == "--version") {
        const BuildInfo b = query_build();
        std::cout << "monkey_bench " << b.version << " (" << b.git_sha << "), suite v" << kVersion << "\n";
        return 0;
    }
    for (const auto& a : args)
        if (a == "-h" || a == "--help") {
            print_help();
            return 0;
        }
    if (args.empty()) return cmd_run({});
    const std::string cmd = args[0];
    const std::vector<std::string> rest(args.begin() + 1, args.end());
    if (cmd == "run") return cmd_run(rest);
    if (cmd == "list") return cmd_list();
    if (cmd == "info") return cmd_info(rest);
    if (cmd == "verify") return cmd_verify(rest);
    if (cmd == "custom") return cmd_custom(rest);
    // Опции run без подкоманды: `monkey_bench --quick`.
    if (cmd == "--quick" || cmd == "--standard" || cmd == "--stress" || cmd == "--cpu" || cmd == "--gpu" ||
        cmd == "--all")
        return cmd_run(args);
    // Старый синтаксис: `monkey_bench <reference_file> -cpu -d 10`, `monkey_bench -w bbp`.
    return cmd_custom(args);
}

} // namespace monkey

int main(int argc, char** argv) {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif
    return monkey::cli(argc, argv);
}
