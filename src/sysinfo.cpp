#include "sysinfo.h"

#include "build_info.h" // сгенерирован CMake: версия, git sha, флаги

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <thread>
#include <utility>

#if defined(__APPLE__)
#include <sys/sysctl.h>
#include <sys/utsname.h>
#elif defined(__linux__)
#include <sys/utsname.h>
#include <unistd.h>
#elif defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <intrin.h>
#endif

namespace monkey {

namespace {

std::string trim(std::string s) {
    const auto b = s.find_first_not_of(" \t\r\n");
    if (b == std::string::npos) return {};
    const auto e = s.find_last_not_of(" \t\r\n");
    return s.substr(b, e - b + 1);
}

#if defined(__APPLE__)

std::string sysctl_str(const char* name) {
    size_t len = 0;
    if (sysctlbyname(name, nullptr, &len, nullptr, 0) != 0 || len == 0) return {};
    std::string s(len, '\0');
    if (sysctlbyname(name, s.data(), &len, nullptr, 0) != 0) return {};
    s.resize(len > 0 && s[len - 1] == '\0' ? len - 1 : len);
    return trim(s);
}

template <class T>
T sysctl_num(const char* name) {
    T v{};
    size_t len = sizeof v;
    if (sysctlbyname(name, &v, &len, nullptr, 0) != 0) return T{};
    return v;
}

#elif defined(__linux__)

std::string read_file(const char* path) {
    std::ifstream f(path);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// Модели ARM-ядер по (implementer, part) из /proc/cpuinfo — самые частые.
std::string arm_core_name(unsigned impl, unsigned part) {
    static const std::map<std::pair<unsigned, unsigned>, const char*> names = {
        {{0x41, 0xd03}, "Cortex-A53"},   {{0x41, 0xd07}, "Cortex-A57"},
        {{0x41, 0xd08}, "Cortex-A72"},   {{0x41, 0xd0b}, "Cortex-A76"},
        {{0x41, 0xd0c}, "Neoverse-N1"},  {{0x41, 0xd40}, "Neoverse-V1"},
        {{0x41, 0xd49}, "Neoverse-N2"},  {{0x41, 0xd4f}, "Neoverse-V2"},
        {{0x41, 0xd8e}, "Neoverse-N3"},  {{0x41, 0xd84}, "Neoverse-V3"},
        {{0x41, 0xd41}, "Cortex-A78"},   {{0x41, 0xd44}, "Cortex-X1"},
        {{0x61, 0x022}, "Apple M1"},     {{0x61, 0x023}, "Apple M1"},
        {{0xc0, 0xac3}, "Ampere-1"},     {{0x48, 0xd01}, "Kunpeng-920"},
    };
    const auto it = names.find({impl, part});
    char buf[64];
    if (it != names.end()) return it->second;
    std::snprintf(buf, sizeof buf, "ARM implementer 0x%x part 0x%x", impl, part);
    return buf;
}

#endif

} // namespace

SystemInfo query_system() {
    SystemInfo si;
    si.cores_logical = std::thread::hardware_concurrency();
#if defined(__aarch64__) || defined(_M_ARM64)
    si.arch = "arm64";
#elif defined(__x86_64__) || defined(_M_X64)
    si.arch = "x86_64";
#else
    si.arch = "unknown";
#endif

#if defined(__APPLE__)
    si.os = "macOS";
    si.os_version = sysctl_str("kern.osproductversion");
    si.kernel = sysctl_str("kern.osrelease");
    si.cpu_model = sysctl_str("machdep.cpu.brand_string");
    si.cores_physical = sysctl_num<int>("hw.physicalcpu");
    si.cores_logical = sysctl_num<int>("hw.logicalcpu");
    // Apple Silicon: perflevel0 — P-ядра, perflevel1 — E-ядра.
    if (sysctl_num<int>("hw.nperflevels") >= 2) {
        si.cores_performance = sysctl_num<int>("hw.perflevel0.physicalcpu");
        si.cores_efficiency = sysctl_num<int>("hw.perflevel1.physicalcpu");
    }
    si.ram_bytes = sysctl_num<uint64_t>("hw.memsize");
#elif defined(__linux__)
    si.os = "Linux";
    {
        std::istringstream rel(read_file("/etc/os-release"));
        std::string line;
        while (std::getline(rel, line)) {
            if (line.rfind("PRETTY_NAME=", 0) == 0) {
                si.os_version = trim(line.substr(12));
                if (si.os_version.size() >= 2 && si.os_version.front() == '"')
                    si.os_version = si.os_version.substr(1, si.os_version.size() - 2);
            }
        }
    }
    utsname u{};
    if (uname(&u) == 0) si.kernel = u.release;
    // Модель и топология из /proc/cpuinfo: физическое ядро = (physical id, core id).
    std::istringstream info(read_file("/proc/cpuinfo"));
    std::string line;
    std::set<std::pair<int, int>> cores;
    int phys = 0;
    unsigned impl = 0, part = 0;
    std::map<std::pair<unsigned, unsigned>, unsigned> arm_count;
    while (std::getline(info, line)) {
        const auto colon = line.find(':');
        if (colon == std::string::npos) continue;
        const std::string key = trim(line.substr(0, colon)), val = trim(line.substr(colon + 1));
        if (key == "model name" && si.cpu_model.empty()) si.cpu_model = val;
        else if (key == "physical id") phys = std::stoi(val);
        else if (key == "core id") cores.insert({phys, std::stoi(val)});
        else if (key == "CPU implementer") impl = static_cast<unsigned>(std::stoul(val, nullptr, 16));
        else if (key == "CPU part") {
            part = static_cast<unsigned>(std::stoul(val, nullptr, 16));
            ++arm_count[{impl, part}];
        }
    }
    if (si.cpu_model.empty() && !arm_count.empty()) {
        // big.LITTLE: перечисляем типы ядер.
        for (const auto& [ip, n] : arm_count) {
            if (!si.cpu_model.empty()) si.cpu_model += " + ";
            si.cpu_model += std::to_string(n) + "x " + arm_core_name(ip.first, ip.second);
        }
    }
    si.cores_physical = cores.empty() ? si.cores_logical : static_cast<unsigned>(cores.size());
    // Гибриды Intel: /sys/devices/cpu_core и cpu_atom перечисляют свои CPU.
    auto count_list = [](const std::string& s) {
        unsigned n = 0;
        std::stringstream ss(s);
        std::string part_s;
        while (std::getline(ss, part_s, ',')) {
            const auto dash = part_s.find('-');
            if (dash == std::string::npos) ++n;
            else n += static_cast<unsigned>(std::stoi(part_s.substr(dash + 1)) - std::stoi(part_s.substr(0, dash)) + 1);
        }
        return n;
    };
    const std::string pcores = trim(read_file("/sys/devices/cpu_core/cpus"));
    const std::string ecores = trim(read_file("/sys/devices/cpu_atom/cpus"));
    if (!pcores.empty() && !ecores.empty()) {
        si.cores_performance = count_list(pcores); // логические P-потоки
        si.cores_efficiency = count_list(ecores);
    }
    const long pages = sysconf(_SC_PHYS_PAGES), page = sysconf(_SC_PAGE_SIZE);
    if (pages > 0 && page > 0) si.ram_bytes = static_cast<uint64_t>(pages) * static_cast<uint64_t>(page);
#elif defined(_WIN32)
    si.os = "Windows";
    // Версия ядра — из RtlGetVersion (GetVersionEx врёт без манифеста).
    using RtlGetVersionFn = LONG(WINAPI*)(OSVERSIONINFOW*);
    if (HMODULE ntdll = GetModuleHandleW(L"ntdll.dll")) {
        if (auto fn = reinterpret_cast<RtlGetVersionFn>(GetProcAddress(ntdll, "RtlGetVersion"))) {
            OSVERSIONINFOW v{};
            v.dwOSVersionInfoSize = sizeof v;
            if (fn(&v) == 0) {
                si.os_version = std::to_string(v.dwMajorVersion) + "." + std::to_string(v.dwMinorVersion) + "." +
                                std::to_string(v.dwBuildNumber);
                si.kernel = std::to_string(v.dwBuildNumber);
            }
        }
    }
#if defined(_M_X64)
    int regs[4];
    char brand[49] = {};
    __cpuid(regs, 0x80000000);
    if (static_cast<unsigned>(regs[0]) >= 0x80000004u) {
        for (int i = 0; i < 3; ++i) {
            __cpuid(regs, 0x80000002 + i);
            std::memcpy(brand + 16 * i, regs, 16);
        }
        si.cpu_model = trim(brand);
    }
#endif
    if (si.cpu_model.empty()) {
        char buf[256];
        DWORD len = sizeof buf;
        if (RegGetValueA(HKEY_LOCAL_MACHINE, "HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0",
                         "ProcessorNameString", RRF_RT_REG_SZ, nullptr, buf, &len) == ERROR_SUCCESS)
            si.cpu_model = trim(buf);
    }
    // Физические ядра и классы эффективности (гибриды Intel 12+ поколения).
    DWORD bytes = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &bytes);
    std::string buf(bytes, '\0');
    if (bytes && GetLogicalProcessorInformationEx(RelationProcessorCore,
                                                  reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buf.data()),
                                                  &bytes)) {
        unsigned phys = 0, max_class = 0;
        std::map<unsigned, unsigned> by_class;
        for (DWORD off = 0; off < bytes;) {
            auto* p = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buf.data() + off);
            ++phys;
            ++by_class[p->Processor.EfficiencyClass];
            max_class = std::max<unsigned>(max_class, p->Processor.EfficiencyClass);
            off += p->Size;
        }
        si.cores_physical = phys;
        if (by_class.size() >= 2) {
            si.cores_performance = by_class[max_class]; // больший класс — производительнее
            si.cores_efficiency = phys - si.cores_performance;
        }
    }
    MEMORYSTATUSEX ms{};
    ms.dwLength = sizeof ms;
    if (GlobalMemoryStatusEx(&ms)) si.ram_bytes = ms.ullTotalPhys;
#endif
    return si;
}

BuildInfo query_build() {
    BuildInfo b;
    b.version = MONKEY_VERSION;
    b.git_sha = MONKEY_GIT_SHA;
    b.flags = MONKEY_BUILD_FLAGS;
    b.build_type = MONKEY_BUILD_TYPE;
#if defined(__clang__)
    b.compiler = std::string("Clang ") + __clang_version__;
#elif defined(__GNUC__)
    b.compiler = "GCC " + std::to_string(__GNUC__) + "." + std::to_string(__GNUC_MINOR__) + "." +
                 std::to_string(__GNUC_PATCHLEVEL__);
#elif defined(_MSC_VER)
    b.compiler = "MSVC " + std::to_string(_MSC_FULL_VER);
#else
    b.compiler = "unknown";
#endif
#if !defined(NDEBUG)
    b.debug = true;
#endif
#if defined(MONKEY_NATIVE_ARCH_ENABLED)
    b.native_arch = true;
#endif
#if defined(MONKEY_PORTABLE_ARITH)
    b.portable_arith = true;
#endif
    return b;
}

} // namespace monkey
