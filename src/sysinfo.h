#pragma once

#include <cstdint>
#include <string>

// Сведения о системе и сборке для отчёта (JSON v2, `info`). Всё, что ОС не
// сообщает, остаётся пустым или нулём — в JSON это null.

namespace monkey {

struct SystemInfo {
    std::string os;          // "macOS", "Linux", "Windows"
    std::string os_version;  // "27.0", "Ubuntu 24.04.1 LTS", "10.0.26100"
    std::string kernel;      // uname -r / build number
    std::string arch;        // "arm64", "x86_64"
    std::string cpu_model;
    unsigned cores_physical = 0;
    unsigned cores_logical = 0;
    unsigned cores_performance = 0; // гибридные CPU: P-ядра (0 — не гибрид/неизвестно)
    unsigned cores_efficiency = 0;  // E-ядра
    uint64_t ram_bytes = 0;
};

SystemInfo query_system();

struct BuildInfo {
    std::string version;
    std::string git_sha;
    std::string compiler;
    std::string flags;
    std::string build_type;
    bool debug = false;          // сборка без оптимизаций/с assert'ами
    bool native_arch = false;    // -march=native
    bool portable_arith = false; // MONKEY_PORTABLE_ARITH
};

BuildInfo query_build();

} // namespace monkey
