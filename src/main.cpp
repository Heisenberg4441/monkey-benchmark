#include "backend.h"
#include "kernels.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#endif

namespace monkey {

std::vector<std::string> split_utf8(const std::string& str) {
    std::vector<std::string> chars;
    size_t i = 0;
    while (i < str.size()) {
        int len = 1;
        unsigned char c = static_cast<unsigned char>(str[i]);
        if ((c & 0x80) == 0) len = 1;
        else if ((c & 0xE0) == 0xC0) len = 2;
        else if ((c & 0xF0) == 0xE0) len = 3;
        else if ((c & 0xF8) == 0xF0) len = 4;
        chars.push_back(str.substr(i, len));
        i += len;
    }
    return chars;
}

namespace {

enum class GpuApi { Auto, Cuda, Vulkan };

enum class OutputFormat { Text, Json };

struct Args {
    std::string file;
    Mode mode = Mode::Random;
    Backend backend = Backend::Cpu;
    GpuApi gpu_api = GpuApi::Auto;
    unsigned threads = 0;
    double duration = 0.0;
    double warmup = 2.0;
    uint32_t seed = 0x9e3779b9u;
    bool seed_set = false;
    uint64_t batch_size = 8192;
    WorkloadType workload = WorkloadType::Monkey;
    OutputFormat output_format = OutputFormat::Text;
    bool vk_allow_cpu = false;
};

void print_help(const char* prog) {
    std::cout <<
        "Infinite Monkey Benchmark\n\n"
        "Использование:\n  " << prog << " [OPTIONS] <reference_file>\n\n"
        "Опции:\n"
        "  -w, --workload <monkey|bbp|miller-rabin>  что считать (default: monkey)\n"
        "  -m, --mode <random|brute>     режим работы (default: random)\n"
        "  -b, --backend <cpu|gpu|all>   целевая нагрузка (default: cpu)\n"
        "      -cpu | -gpu | -all        короткие алиасы для --backend\n"
        "      --gpu-api <auto|cuda|vulkan>  выбор GPU-API (default: auto)\n"
        "      --vk-allow-cpu            разрешить программный Vulkan (lavapipe)\n"
        "  -t, --threads <N>             число CPU-потоков (default: auto)\n"
        "  -d, --duration <sec>          остановиться через N секунд (бенчмарк)\n"
        "      --warmup <sec>            время стабилизации перед замером (default: 2)\n"
        "      --output-format <text|json>  вывод в JSON для CI (default: text)\n"
        "      --seed <N>                seed counter-based PRNG (воспроизводимость)\n"
        "      --batch-size <N>          гранулярность синка счётчика (default: 8192)\n"
        "  -h, --help                    показать эту справку\n";
}

bool take_value(int argc, char** argv, int& i, const std::string& arg,
                std::string& out) {
    auto eq = arg.find('=');
    if (eq != std::string::npos) { out = arg.substr(eq + 1); return true; }
    if (i + 1 < argc) { out = argv[++i]; return true; }
    return false;
}

bool parse_backend(const std::string& v, Backend& out) {
    if (v == "cpu") out = Backend::Cpu;
    else if (v == "gpu") out = Backend::Gpu;
    else if (v == "all") out = Backend::All;
    else return false;
    return true;
}

bool parse_args(int argc, char** argv, Args& args) {
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        std::string key = a.substr(0, a.find('='));

        if (a == "-h" || a == "--help") { print_help(argv[0]); std::exit(0); }
        else if (a == "-cpu") args.backend = Backend::Cpu;
        else if (a == "-gpu") args.backend = Backend::Gpu;
        else if (a == "-all") args.backend = Backend::All;
        else if (key == "-w" || key == "--workload") {
            std::string v;
            if (!take_value(argc, argv, i, a, v)) { std::cerr << "Нет значения для " << key << "\n"; return false; }
            if (v == "monkey") args.workload = WorkloadType::Monkey;
            else if (v == "bbp") args.workload = WorkloadType::Bbp;
            else if (v == "miller-rabin") args.workload = WorkloadType::MillerRabin;
            else { std::cerr << "Неизвестный workload: " << v << "\n"; return false; }
        }
        else if (key == "-m" || key == "--mode") {
            std::string v;
            if (!take_value(argc, argv, i, a, v)) { std::cerr << "Нет значения для " << key << "\n"; return false; }
            if (v == "random") args.mode = Mode::Random;
            else if (v == "brute") args.mode = Mode::Brute;
            else { std::cerr << "Неизвестный режим: " << v << "\n"; return false; }
        }
        else if (key == "-b" || key == "--backend") {
            std::string v;
            if (!take_value(argc, argv, i, a, v)) { std::cerr << "Нет значения для " << key << "\n"; return false; }
            if (!parse_backend(v, args.backend)) { std::cerr << "Неизвестный бэкенд: " << v << "\n"; return false; }
        }
        else if (key == "--gpu-api") {
            std::string v;
            if (!take_value(argc, argv, i, a, v)) { std::cerr << "Нет значения для " << key << "\n"; return false; }
            if (v == "auto") args.gpu_api = GpuApi::Auto;
            else if (v == "cuda") args.gpu_api = GpuApi::Cuda;
            else if (v == "vulkan") args.gpu_api = GpuApi::Vulkan;
            else { std::cerr << "Неизвестный gpu-api: " << v << "\n"; return false; }
        }
        else if (key == "-t" || key == "--threads") {
            std::string v;
            if (!take_value(argc, argv, i, a, v)) { std::cerr << "Нет значения для " << key << "\n"; return false; }
            args.threads = static_cast<unsigned>(std::strtoul(v.c_str(), nullptr, 10));
        }
        else if (key == "-d" || key == "--duration") {
            std::string v;
            if (!take_value(argc, argv, i, a, v)) { std::cerr << "Нет значения для " << key << "\n"; return false; }
            args.duration = std::strtod(v.c_str(), nullptr);
        }
        else if (key == "--warmup") {
            std::string v;
            if (!take_value(argc, argv, i, a, v)) { std::cerr << "Нет значения для " << key << "\n"; return false; }
            args.warmup = std::strtod(v.c_str(), nullptr);
        }
        else if (key == "--output-format") {
            std::string v;
            if (!take_value(argc, argv, i, a, v)) { std::cerr << "Нет значения для " << key << "\n"; return false; }
            if (v == "json") args.output_format = OutputFormat::Json;
            else if (v == "text") args.output_format = OutputFormat::Text;
            else { std::cerr << "Неизвестный формат вывода: " << v << "\n"; return false; }
        }
        else if (a == "--vk-allow-cpu") args.vk_allow_cpu = true;
        else if (key == "--seed") {
            std::string v;
            if (!take_value(argc, argv, i, a, v)) { std::cerr << "Нет значения для " << key << "\n"; return false; }
            args.seed = static_cast<uint32_t>(std::strtoul(v.c_str(), nullptr, 0));
            args.seed_set = true;
        }
        else if (key == "--batch-size") {
            std::string v;
            if (!take_value(argc, argv, i, a, v)) { std::cerr << "Нет значения для " << key << "\n"; return false; }
            args.batch_size = std::strtoull(v.c_str(), nullptr, 10);
            if (args.batch_size == 0) { std::cerr << "--batch-size должен быть > 0\n"; return false; }
        }
        else if (!a.empty() && a[0] == '-') {
            std::cerr << "Неизвестная опция: " << a << "\n";
            return false;
        }
        else {
            if (!args.file.empty()) { std::cerr << "Лишний аргумент: " << a << "\n"; return false; }
            args.file = a;
        }
    }
    if (args.file.empty()) { std::cerr << "Не указан reference_file. См. --help\n"; return false; }
    return true;
}

const char* mode_name(Mode m) { return m == Mode::Random ? "random" : "brute"; }
const char* workload_name(WorkloadType t) {
    switch (t) {
    case WorkloadType::Monkey: return "monkey";
    case WorkloadType::Bbp: return "bbp";
    case WorkloadType::MillerRabin: return "miller-rabin";
    }
    return "unknown";
}
const char* backend_name(Backend b) {
    return b == Backend::Cpu ? "cpu" : (b == Backend::Gpu ? "gpu" : "all");
}

// Один сэмпл Ops/sec: момент времени + мгновенная скорость.
struct Sample {
    double elapsed;
    double ops_per_sec;
};

// Статистики по набору сэмплов.
struct Stats {
    double min = 0;
    double median = 0;
    double mean = 0;
    double p95 = 0;
    double p99 = 0;
    double max = 0;
    double stddev = 0;
    double cv = 0; // coefficient of variation: stddev / mean
};

Stats compute_stats(std::vector<Sample>& samples) {
    if (samples.empty()) return {};
    const size_t n = samples.size();

    std::sort(samples.begin(), samples.end(),
              [](const Sample& a, const Sample& b) { return a.ops_per_sec < b.ops_per_sec; });

    Stats s;
    s.min = samples.front().ops_per_sec;
    s.max = samples.back().ops_per_sec;

    const size_t p95_idx = static_cast<size_t>(n * 0.95);
    const size_t p99_idx = static_cast<size_t>(n * 0.99);
    s.p95 = samples[p95_idx < n ? p95_idx : n - 1].ops_per_sec;
    s.p99 = samples[p99_idx < n ? p99_idx : n - 1].ops_per_sec;
    s.median = samples[n / 2].ops_per_sec;

    double sum = 0;
    for (const auto& sp : samples) sum += sp.ops_per_sec;
    s.mean = sum / static_cast<double>(n);

    double sq = 0;
    for (const auto& sp : samples) {
        const double d = sp.ops_per_sec - s.mean;
        sq += d * d;
    }
    s.stddev = std::sqrt(sq / static_cast<double>(n));
    s.cv = s.mean > 0 ? s.stddev / s.mean : 0;

    return s;
}

void print_stats(const Stats& s) {
    std::cout << "\n--- Статистика Ops/sec (" << s.mean << ") ---\n";
    std::cout << "  min:    " << static_cast<long long>(s.min) << "\n";
    std::cout << "  median: " << static_cast<long long>(s.median) << "\n";
    std::cout << "  mean:   " << static_cast<long long>(s.mean) << "\n";
    std::cout << "  p95:    " << static_cast<long long>(s.p95) << "\n";
    std::cout << "  p99:    " << static_cast<long long>(s.p99) << "\n";
    std::cout << "  max:    " << static_cast<long long>(s.max) << "\n";
    std::cout << "  stddev: " << static_cast<long long>(s.stddev) << "\n";
    std::cout << "  CV:     " << (s.cv * 100.0) << "%\n";
    if (s.cv > 0.05) {
        std::cerr << "[warn] CV > 5% — измерение нестабильно (термал/фон?)\n";
    }
}

void print_json(const std::string& reference, const Config& cfg, const char* gpu_label,
                bool use_cpu, bool use_gpu, double total_time, uint64_t total_attempts,
                const Stats& s, bool solution_found) {
    // Minimal JSON string escaping.
    auto json_escape = [](const std::string& s) -> std::string {
        std::string out;
        out.reserve(s.size());
        for (char c : s) {
            switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default: out += c;
            }
        }
        return out;
    };
    std::cout << "{\n";
    std::cout << "  \"reference\": \"" << json_escape(reference) << "\",\n";
    std::cout << "  \"alphabet_size\": " << cfg.n << ",\n";
    std::cout << "  \"string_length\": " << cfg.len << ",\n";
    std::cout << "  \"workload\": \"" << workload_name(cfg.workload) << "\",\n";
    std::cout << "  \"mode\": \"" << mode_name(cfg.mode) << "\",\n";
    std::cout << "  \"backend\": \""
              << (use_cpu && use_gpu ? std::string("cpu+") + gpu_label
                                    : (use_gpu ? gpu_label : "cpu"))
              << "\",\n";
    std::cout << "  \"threads\": " << (cfg.threads > 0 ? cfg.threads : std::thread::hardware_concurrency()) << ",\n";
    std::cout << "  \"batch_size\": " << cfg.batch_size << ",\n";
    std::cout << "  \"seed\": " << cfg.seed << ",\n";
    std::cout << "  \"total_time_s\": " << total_time << ",\n";
    std::cout << "  \"total_attempts\": " << total_attempts << ",\n";
    std::cout << "  \"solution_found\": " << (solution_found ? "true" : "false") << ",\n";
    std::cout << "  \"ops_per_sec\": {\n";
    std::cout << "    \"min\": " << static_cast<long long>(s.min) << ",\n";
    std::cout << "    \"median\": " << static_cast<long long>(s.median) << ",\n";
    std::cout << "    \"mean\": " << static_cast<long long>(s.mean) << ",\n";
    std::cout << "    \"p95\": " << static_cast<long long>(s.p95) << ",\n";
    std::cout << "    \"p99\": " << static_cast<long long>(s.p99) << ",\n";
    std::cout << "    \"max\": " << static_cast<long long>(s.max) << ",\n";
    std::cout << "    \"stddev\": " << static_cast<long long>(s.stddev) << ",\n";
    std::cout << "    \"cv\": " << s.cv << "\n";
    std::cout << "  }\n";
    std::cout << "}\n";
}

} // namespace

int run(int argc, char** argv) {
    Args args;
    if (!parse_args(argc, argv, args)) return 1;

    std::ifstream file(args.file);
    if (!file.is_open()) {
        std::cerr << "Ошибка: не удалось открыть " << args.file << "\n";
        return 1;
    }
    std::string reference((std::istreambuf_iterator<char>(file)),
                          std::istreambuf_iterator<char>());
    file.close();
    while (!reference.empty() && (reference.back() == '\n' || reference.back() == '\r'))
        reference.pop_back();
    if (reference.empty()) { std::cerr << "Ошибка: файл пуст!\n"; return 1; }

    std::vector<std::string> target_chars = split_utf8(reference);
    std::vector<std::string> alphabet = target_chars;
    std::sort(alphabet.begin(), alphabet.end());
    alphabet.erase(std::unique(alphabet.begin(), alphabet.end()), alphabet.end());

    std::map<std::string, int> index;
    for (size_t i = 0; i < alphabet.size(); ++i) index[alphabet[i]] = static_cast<int>(i);

    Config cfg;
    cfg.alphabet = alphabet;
    cfg.n = static_cast<int>(alphabet.size());
    cfg.len = static_cast<int>(target_chars.size());
    cfg.target_idx.reserve(target_chars.size());
    for (const auto& ch : target_chars) cfg.target_idx.push_back(index[ch]);
    cfg.mode = args.mode;
    cfg.backend = args.backend;
    cfg.threads = args.threads;
    cfg.duration = args.duration;
    cfg.seed = args.seed;
    cfg.batch_size = args.batch_size;
    cfg.workload = args.workload;

    // Что исполняют бэкенды. BBP и MR — фиксированные окна suite (стоимость
    // операции не растёт со временем прогона), monkey — эталон из файла.
    KernelParams& kp = cfg.kernel;
    switch (cfg.workload) {
    case WorkloadType::Monkey:
        kp.kernel = (cfg.mode == Mode::Brute) ? KernelId::MonkeyBrute : KernelId::MonkeyRandom;
        kp.seed = cfg.seed;
        kp.n = static_cast<uint32_t>(cfg.n);
        kp.target.assign(cfg.target_idx.begin(), cfg.target_idx.end());
        break;
    case WorkloadType::Bbp:
        kp.kernel = KernelId::Bbp;
        kp.bbp = bbp_window(1024);
        break;
    case WorkloadType::MillerRabin:
        kp.kernel = KernelId::MillerRabin;
        kp.mr_base = (1ull << 62) + 1;
        kp.mr_step = 2;
        break;
    }
    if (kp.kernel == KernelId::MonkeyBrute) {
        // Индекс 64-битный: эталон с кодом >= 2^64 недостижим перебором. С -d это
        // просто замер скорости перебора, без -d — бесконечный поиск, ошибка.
        uint64_t code = 0;
        for (int32_t t : kp.target) {
            if (code > (UINT64_MAX - static_cast<uint64_t>(t)) / kp.n) {
                if (cfg.duration <= 0) {
                    std::cerr << "Ошибка: brute — код эталона не помещается в 64 бита "
                                 "(n^len > 2^64), совпадение недостижимо; задайте -d или "
                                 "используйте random\n";
                    return 1;
                }
                std::cerr << "[warn] brute: n^len > 2^64, совпадение недостижимо — только замер\n";
                break;
            }
            code = code * kp.n + static_cast<uint64_t>(t);
        }
    }
    {
        std::string why;
        if (!kernel_selfcheck(kp, &why)) {
            std::cerr << "Ошибка: самопроверка ядра '" << kernel_name(kp.kernel)
                      << "' не прошла (" << why << ") — бенч не запущен\n";
            return 1;
        }
    }

    bool use_cpu = (cfg.backend == Backend::Cpu || cfg.backend == Backend::All);
    bool use_gpu = (cfg.backend == Backend::Gpu || cfg.backend == Backend::All);

    // GPU создаётся один раз и переживает прогрев: инициализация (instance,
    // pipeline'ы, контекст) не попадает в окно замера. Явно запрошенный, но
    // недоступный GPU — ошибка, а не тихий откат на CPU.
    std::unique_ptr<GpuDevice> gpu;
    const char* gpu_label = "";
    if (use_gpu) {
        const bool want_cuda = (args.gpu_api == GpuApi::Auto || args.gpu_api == GpuApi::Cuda);
        const bool want_vulkan = (args.gpu_api == GpuApi::Auto || args.gpu_api == GpuApi::Vulkan);
        std::string err_cuda, err_vk;
        if (want_cuda) gpu = create_cuda_device(CudaOptions{}, &err_cuda);
        if (!gpu && want_vulkan) {
            VulkanOptions vo;
            vo.allow_cpu_device = args.vk_allow_cpu;
            gpu = create_vulkan_device(vo, &err_vk);
        }
        if (!gpu) {
            std::cerr << "Ошибка: GPU недоступен";
            if (!err_cuda.empty()) std::cerr << "\n  " << err_cuda;
            if (!err_vk.empty()) std::cerr << "\n  " << err_vk;
            std::cerr << "\n";
            return 1;
        }
        gpu_label = gpu->info().api == "cuda" ? "cuda" : "vulkan";
        std::cerr << "[" << gpu_label << "] устройство: " << gpu->info().name << " | init "
                  << gpu->info().init_time_s << " c";
        if (!gpu->info().u64_mode.empty()) std::cerr << " | u64: " << gpu->info().u64_mode;
        std::cerr << "\n";
    }

    if (args.output_format == OutputFormat::Text) {
        std::cout << "Эталон:           " << reference << "\n";
        std::cout << "Длина (символов): " << cfg.len << "\n";
        std::cout << "Размер алфавита:  " << cfg.n << "\n";
        std::cout << "Workload:         " << workload_name(cfg.workload) << "\n";
        std::cout << "Режим:            " << mode_name(cfg.mode) << "\n";
        std::cout << "Бэкенд:           " << backend_name(cfg.backend)
                  << (use_cpu && use_gpu ? std::string(" (cpu+") + gpu_label + ")"
                                         : (use_gpu ? std::string(" (") + gpu_label + ")" : std::string(" (cpu)"))) << "\n";
        if (args.seed_set) std::cout << "Seed PRNG:        " << cfg.seed << "\n";
        if (cfg.duration > 0) std::cout << "Лимит времени:    " << cfg.duration << " c\n";
        if (args.warmup > 0) std::cout << "Прогрев:          " << args.warmup << " c\n";
        std::cout << "\n";
    }

    // --- Warmup фиксация измерений ------------------------------------------
    if (args.warmup > 0) {
        Control warm_ctrl;
        auto warm_start = std::chrono::high_resolution_clock::now();

        std::thread w_cpu, w_gpu;
        if (use_cpu) w_cpu = std::thread(run_cpu, std::cref(cfg), std::ref(warm_ctrl));
        if (use_gpu) w_gpu = std::thread(run_gpu, std::ref(*gpu), std::cref(cfg), std::ref(warm_ctrl));

        while (!warm_ctrl.stop.load(std::memory_order_acquire)) {
            double elapsed = std::chrono::duration<double>(
                std::chrono::high_resolution_clock::now() - warm_start).count();
            if (elapsed >= args.warmup)
                warm_ctrl.stop.store(true, std::memory_order_release);
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        if (w_cpu.joinable()) w_cpu.join();
        if (w_gpu.joinable()) w_gpu.join();
        if (warm_ctrl.failed.load()) {
            std::cerr << "Ошибка: бэкенд упал на прогреве — бенч остановлен\n";
            return 1;
        }
    }

    // --- Измерение --------------------------------------------------------
    Control ctrl;
    auto measure_start = std::chrono::high_resolution_clock::now();

    std::thread cpu_thr, gpu_thr;
    if (use_cpu) cpu_thr = std::thread(run_cpu, std::cref(cfg), std::ref(ctrl));
    if (use_gpu) gpu_thr = std::thread(run_gpu, std::ref(*gpu), std::cref(cfg), std::ref(ctrl));

    // Сэмплирование Ops/sec каждые ~100 мс.
    constexpr auto kSampleInterval = std::chrono::milliseconds(100);
    std::vector<Sample> samples;
    unsigned long long prev_total = 0;
    auto prev_sample_time = measure_start;

    while (!ctrl.stop.load(std::memory_order_acquire)) {
        std::this_thread::sleep_for(kSampleInterval);
        auto now = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(now - measure_start).count();
        unsigned long long cpu_a = ctrl.cpu_counters.total();
        unsigned long long gpu_a = ctrl.gpu_attempts.load(std::memory_order_relaxed);
        unsigned long long total = cpu_a + gpu_a;

        double sample_elapsed = std::chrono::duration<double>(now - prev_sample_time).count();
        if (sample_elapsed > 0) {
            long long sample_attempts = static_cast<long long>(total - prev_total);
            double rate = static_cast<double>(sample_attempts) / sample_elapsed;
            samples.push_back({elapsed, rate});
        }
        prev_total = total;
        prev_sample_time = now;

        if (args.output_format == OutputFormat::Text) {
            long long avg_rate = elapsed > 0 ? static_cast<long long>(total / elapsed) : 0;
            std::cout << "\rПрошло: " << elapsed << "c | Попыток: " << total
                      << " | Скорость: " << avg_rate << " оп/с";
            if (use_cpu && use_gpu)
                std::cout << " (cpu " << cpu_a << " / gpu " << gpu_a << ")";
            std::cout << "          " << std::flush;
        }

        if (cfg.duration > 0 && elapsed >= cfg.duration)
            ctrl.stop.store(true, std::memory_order_release);
    }

    if (cpu_thr.joinable()) cpu_thr.join();
    if (gpu_thr.joinable()) gpu_thr.join();

    if (ctrl.failed.load()) {
        std::cerr << "\nОшибка: бэкенд упал во время замера — результат недействителен\n";
        return 1;
    }

    double total_time = std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now() - measure_start).count();
    uint64_t total_attempts = ctrl.cpu_counters.total() + ctrl.gpu_attempts.load();
    bool solution_found = ctrl.found.load();

    Stats stats = compute_stats(samples);

    if (args.output_format == OutputFormat::Json) {
        print_json(reference, cfg, gpu_label, use_cpu, use_gpu,
                   total_time, total_attempts, stats, solution_found);
    } else {
        std::cout << "\n\n";
        if (solution_found) std::cout << "=== СОВПАДЕНИЕ НАЙДЕНО ===\n";
        else std::cout << "=== ОСТАНОВКА ПО ЛИМИТУ ВРЕМЕНИ ===\n";
        std::cout << "Время:           " << total_time << " c\n";
        std::cout << "Всего попыток:   " << total_attempts << "\n";
        std::cout << "Средняя скорость: "
                  << (total_time > 0 ? static_cast<long long>(total_attempts / total_time) : 0)
                  << " оп/с\n";
        if (!samples.empty()) print_stats(stats);
    }

    return 0;
}

} // namespace monkey

int main(int argc, char** argv) {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif
    return monkey::run(argc, argv);
}
