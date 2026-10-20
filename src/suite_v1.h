#pragma once

#include <cstdint>

#include "kernels.h"

// Suite v1 — все константы, от которых зависит результат: состав тестов,
// параметры kernel'ов, объём работы по категориям, эталонные времена и golden.
// Любое изменение ядра, размера или golden поднимает kVersion: баллы разных
// версий не сравниваются. Golden и t_ref генерирует tools/suite_gen.cpp.

namespace monkey {
namespace suite_v1 {

constexpr int kVersion = 1;

// Категории баллов. GPU — одна категория на каждый доступный API (CUDA, Vulkan)
// с одинаковыми размерами и t_ref.
enum Category : int { kCpuSt = 0, kCpuMt = 1, kGpu = 2, kCategories = 3 };

struct Golden {
    uint64_t checksum;
    uint64_t hits;
};

struct TestDef {
    const char* name;
    const char* about;
    KernelId kernel;
    uint64_t size[kCategories]; // индексов; для bbp-pi — позиций окна
    double t_ref[kCategories];  // время на эталонной машине, с (балл 1000)
    Golden golden[kCategories]; // для bbp-pi — по блокам; цифры сверяются с π
};

// Параметры kernel'ов.
constexpr char kFox[] = "The quick brown fox jumps over the lazy dog"; // n = 28, len = 43
constexpr uint32_t kSeed = 0x9e3779b9u;
constexpr uint32_t kBbpBlockTerms = 256;
constexpr uint64_t kMrBase = (1ull << 62) + 1; // нечётные n = kMrBase + 2i
constexpr uint64_t kMrStep = 2;

// Методика.
constexpr double kUnstableCv = 0.03;   // CV времени по повторам выше — тест «unstable»
constexpr int kRunsStandard = 3;
constexpr int kRunsQuick = 1;
constexpr double kWarmupStandard = 2.0; // с, на каждый тест и категорию
constexpr double kWarmupQuick = 1.0;

// Эталонная машина (балл 1000). t_ref — медиана 5 прогонов `run --standard`
// (простой, питание от сети, режим высокой мощности, пауза 2 мин между
// прогонами), см. BENCH.md.
constexpr char kReferenceMachine[] = "Apple M4 Pro (10P+4E), macOS 27.0, MoltenVK 1.4.2";

// clang-format off
constexpr TestDef kTests[] = {
    {"monkey-random", "Philox4x32-10 candidates for the fox pangram, matched prefix length",
     KernelId::MonkeyRandom,
     {800000000ull, 7600000000ull, 24000000000ull},
     {5.253, 4.927, 4.913},
     {{0x803d74d51ae9cd93ull, 28568179ull}, {0xf31ceddec4340ad2ull, 271441309ull}, {0xa5c39389a94098deull, 857177473ull}}},
    {"monkey-brute", "base-28 enumeration of the fox pangram, matched suffix length",
     KernelId::MonkeyBrute,
     {8600000000ull, 84000000000ull, 52000000000ull},
     {4.883, 4.981, 4.978},
     {{0x53c7c613cdd22e99ull, 307142857ull}, {0xa2c022813043deffull, 3000000000ull}, {0x58de71628b47c298ull, 1857142857ull}}},
    {"bbp-pi", "hex digits of pi near position 10^6, integer BBP, blocks of 256 terms",
     KernelId::Bbp,
     {46ull, 480ull, 2600ull},
     {4.710, 4.835, 4.894},
     {{0x33602ed7ab8ad5f0ull, 179722ull}, {0xa320c9765d929053ull, 1874768ull}, {0x72c2f4b19147ebf2ull, 10144280ull}}},
    {"mr-prime64", "deterministic Miller-Rabin over odd numbers above 2^62",
     KernelId::MillerRabin,
     {19000000ull, 185000000ull, 490000000ull},
     {5.022, 4.944, 5.366},
     {{0xc6f105a011137f99ull, 884708ull}, {0x49b5965daa73e04full, 8608581ull}, {0x874790c631d99c1dull, 22801370ull}}},
};
// clang-format on

constexpr int kTestCount = sizeof(kTests) / sizeof(kTests[0]);

// Короткие диапазоны для `verify`: каждое ядро на каждом бэкенде против golden.
// Достаточно малы для программного Vulkan (lavapipe) в CI.
struct VerifyCase {
    int test;
    Category category; // чьи параметры (окно BBP)
    uint64_t begin;
    uint64_t count;
    Golden golden;
};

// clang-format off
constexpr VerifyCase kVerify[] = {
    {0, kCpuSt, 0, 1ull << 20, {0xefe865070be22081ull, 37532ull}},
    {0, kCpuSt, (1ull << 40) - 12345, 1ull << 20, {0x016ab6faa0f4cdaeull, 37728ull}},
    {1, kCpuSt, 0, 1ull << 20, {0xa9d5f1a50a97d5a1ull, 37449ull}},
    {1, kCpuSt, ~0ull - (1ull << 20), 1ull << 20, {0xbba3fed3a51a943aull, 37449ull}},
    {2, kGpu, 0, 48, {0x6ce55bd627f27184ull, 48ull}},
    {3, kCpuSt, 0, 1ull << 16, {0xfa160ff45187cc6cull, 3114ull}},
};
// clang-format on

// Параметры kernel'а теста в категории.
KernelParams make_params(const TestDef& t, Category c);

// Число индексов работы теста в категории (для bbp-pi — блоков окна).
uint64_t work_units(const TestDef& t, Category c);

const char* category_key(Category c); // "cpu_st", "cpu_mt", "gpu"
const char* category_title(Category c);

} // namespace suite_v1
} // namespace monkey
