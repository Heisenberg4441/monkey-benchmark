#include "backend.h"

#include <algorithm>
#include <chrono>
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

struct Args {
    std::string file;
    Mode mode = Mode::Random;
    Backend backend = Backend::Cpu;
    unsigned threads = 0;
    double duration = 0.0;
};

void print_help(const char* prog) {
    std::cout <<
        "Infinite Monkey Benchmark\n\n"
        "Использование:\n  " << prog << " [OPTIONS] <reference_file>\n\n"
        "Опции:\n"
        "  -m, --mode <random|brute>     режим работы (default: random)\n"
        "  -b, --backend <cpu|gpu|all>   целевая нагрузка (default: cpu)\n"
        "      -cpu | -gpu | -all        короткие алиасы для --backend\n"
        "  -t, --threads <N>             число CPU-потоков (default: auto)\n"
        "  -d, --duration <sec>          остановиться через N секунд (бенчмарк)\n"
        "  -h, --help                    показать эту справку\n";
}

// Возвращает значение опции: либо часть после '=', либо следующий аргумент.
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
const char* backend_name(Backend b) {
    return b == Backend::Cpu ? "cpu" : (b == Backend::Gpu ? "gpu" : "all");
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
    // Отбрасываем завершающий перевод строки, частый артефакт текстовых файлов.
    while (!reference.empty() && (reference.back() == '\n' || reference.back() == '\r'))
        reference.pop_back();
    if (reference.empty()) { std::cerr << "Ошибка: файл пуст!\n"; return 1; }

    std::vector<std::string> target_chars = split_utf8(reference);
    std::vector<std::string> alphabet = target_chars;
    std::sort(alphabet.begin(), alphabet.end());
    alphabet.erase(std::unique(alphabet.begin(), alphabet.end()), alphabet.end());

    // Сворачиваем эталон к индексам символов в алфавите.
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

    // Разрешаем фактически используемые бэкенды с откатом на CPU.
    bool use_cpu = (cfg.backend == Backend::Cpu || cfg.backend == Backend::All);
    bool use_gpu = (cfg.backend == Backend::Gpu || cfg.backend == Backend::All);
    if (use_gpu && !cuda_available()) {
        std::cerr << "[warn] CUDA-бэкенд недоступен (нет GPU или сборка без CUDA), откат на CPU\n";
        use_gpu = false;
        use_cpu = true;
    }

    std::cout << "Эталон:           " << reference << "\n";
    std::cout << "Длина (символов): " << cfg.len << "\n";
    std::cout << "Размер алфавита:  " << cfg.n << "\n";
    std::cout << "Режим:            " << mode_name(cfg.mode) << "\n";
    std::cout << "Бэкенд:           " << backend_name(cfg.backend)
              << (use_cpu && use_gpu ? " (cpu+gpu)" : (use_gpu ? " (gpu)" : " (cpu)")) << "\n";
    if (cfg.duration > 0) std::cout << "Лимит времени:    " << cfg.duration << " c\n";
    std::cout << "\n";

    Control ctrl;
    auto start = std::chrono::high_resolution_clock::now();

    std::thread cpu_thr, gpu_thr;
    if (use_cpu) cpu_thr = std::thread(run_cpu, std::cref(cfg), std::ref(ctrl));
    if (use_gpu) gpu_thr = std::thread(run_cuda, std::cref(cfg), std::ref(ctrl));

    // Репортер в главном потоке.
    while (!ctrl.stop.load(std::memory_order_acquire)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(250));
        double elapsed = std::chrono::duration<double>(
            std::chrono::high_resolution_clock::now() - start).count();
        unsigned long long cpu_a = ctrl.cpu_attempts.load(std::memory_order_relaxed);
        unsigned long long gpu_a = ctrl.gpu_attempts.load(std::memory_order_relaxed);
        unsigned long long total = cpu_a + gpu_a;
        long long rate = elapsed > 0 ? static_cast<long long>(total / elapsed) : 0;
        std::cout << "Прошло: " << elapsed << "c | Попыток: " << total
                  << " | Скорость: " << rate << " оп/с";
        if (use_cpu && use_gpu)
            std::cout << " (cpu " << cpu_a << " / gpu " << gpu_a << ")";
        std::cout << "          \r" << std::flush;

        if (cfg.duration > 0 && elapsed >= cfg.duration)
            ctrl.stop.store(true, std::memory_order_release);
    }

    if (cpu_thr.joinable()) cpu_thr.join();
    if (gpu_thr.joinable()) gpu_thr.join();

    double total_time = std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now() - start).count();
    unsigned long long total = ctrl.cpu_attempts.load() + ctrl.gpu_attempts.load();

    std::cout << "\n\n";
    if (ctrl.found.load()) std::cout << "=== СОВПАДЕНИЕ НАЙДЕНО ===\n";
    else std::cout << "=== ОСТАНОВКА ПО ЛИМИТУ ВРЕМЕНИ ===\n";
    std::cout << "Время:           " << total_time << " c\n";
    std::cout << "Всего попыток:   " << total << "\n";
    std::cout << "Средняя скорость: "
              << (total_time > 0 ? static_cast<long long>(total / total_time) : 0)
              << " оп/с\n";
    return 0;
}

} // namespace monkey

int main(int argc, char** argv) {
#ifdef _WIN32
    // Вывод в UTF-8 без необходимости вручную делать `chcp 65001`.
    SetConsoleOutputCP(CP_UTF8);
#endif
    return monkey::run(argc, argv);
}
