#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "workloads/bbp_kernel.h"

// Описание вычисления, общее для всех бэкендов: какой kernel и с какими
// параметрами. Тест suite — это KernelParams плюс диапазон индексов [0, N).
// Любой бэкенд (CPU, CUDA, Vulkan) обязан для одного и того же диапазона
// вернуть один и тот же RangeResult — это проверяют тесты паритета.

namespace monkey {

enum class KernelId : uint32_t {
    MonkeyRandom = 0, // r = длина совпавшего префикса случайного кандидата
    MonkeyBrute = 1,  // r = длина совпавшего суффикса кандидата «i по основанию n»
    Bbp = 2,          // r = частичная сумма блока членов BBP (mod 2^64)
    MillerRabin = 3,  // r = 1, если mr_base + mr_step·i простое
};

const char* kernel_name(KernelId k);

struct KernelParams {
    KernelId kernel = KernelId::MonkeyRandom;
    // monkey: эталон в индексах алфавита, мощность алфавита, seed Philox.
    uint32_t seed = 0x9e3779b9u;
    uint32_t n = 0;
    std::vector<int32_t> target;
    // bbp: окно позиций и разбиение на блоки.
    kernel::BbpParams bbp{};
    // miller-rabin: число индекса i — mr_base + mr_step·i.
    uint64_t mr_base = 0;
    uint64_t mr_step = 1;

    uint32_t len() const { return static_cast<uint32_t>(target.size()); }
};

// Итог обработки диапазона индексов. Все поля складываются (merge) в любом
// порядке: результат не зависит от разбиения работы.
struct RangeResult {
    uint64_t count = 0;    // обработано индексов
    uint64_t hits = 0;     // индексов с r != 0
    uint64_t checksum = 0; // Σ contrib(i, r) mod 2^64
    // monkey: наименьший индекс с полным совпадением (r == len), иначе UINT64_MAX.
    uint64_t first_full = UINT64_MAX;

    void merge(const RangeResult& o) {
        count += o.count;
        hits += o.hits;
        checksum += o.checksum;
        if (o.first_full < first_full) first_full = o.first_full;
    }
    bool operator==(const RangeResult& o) const {
        return count == o.count && hits == o.hits && checksum == o.checksum &&
               first_full == o.first_full;
    }
};

// Окно BBP по умолчанию: positions последних позиций встроенного pi_hex.txt,
// блоки по block_terms членов. 256 членов — ~25 мкс на ядре CPU и ~4 мс на
// потоке GPU (M4 Pro): GPU выходит на полную скорость уже при диспатче ~30 мс.
kernel::BbpParams bbp_window(uint32_t positions, uint32_t block_terms = 256);

// Число индексов, покрывающих окно BBP (positions · blocks).
uint64_t bbp_units(const kernel::BbpParams& p);

// Эталонная (CPU) обработка индексов [begin, begin + count). bbp_acc — массив
// p.bbp.positions сумм по позициям (только для Bbp, иначе может быть nullptr):
// к acc[w] прибавляется r каждого блока позиции w.
RangeResult cpu_run_range(const KernelParams& p, uint64_t begin, uint64_t count,
                          uint64_t* bbp_acc);

// Итог BBP-окна: полные суммы позиций (блоки + хвост) -> 8 hex-цифр, сверка со
// встроенным π. checksum = Σ contrib(позиция, цифры); expected — то же по файлу.
struct BbpVerdict {
    uint32_t bad_positions = 0;
    uint64_t checksum = 0;
    uint64_t expected = 0;
};
BbpVerdict bbp_verify(const kernel::BbpParams& p, const std::vector<uint64_t>& acc);

// Самопроверка ядра перед запуском: известные ответы, без зависимостей от
// файлов. false — бенч не запускается.
bool kernel_selfcheck(const KernelParams& p, std::string* why);

} // namespace monkey
