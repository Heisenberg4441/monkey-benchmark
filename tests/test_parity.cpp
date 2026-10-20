// Паритет бэкендов: для каждого kernel'а фиксированный диапазон индексов даёт
// на GPU ровно тот же RangeResult (count, hits, checksum, first_full) и те же
// суммы BBP по позициям, что эталонный cpu_run_range. Диапазон режется на
// диспатчи разного размера — результат не должен зависеть от разбиения.
//
// Vulkan проверяется в обоих вариантах SPIR-V (uint64_t и uvec2). Программные
// устройства (lavapipe в CI) разрешены. Нет устройства — WARN и пропуск.
#include "catch.hpp"
#include "gpu_device.h"
#include "kernels.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

using namespace monkey;

namespace {

KernelParams fox(KernelId k) {
    // «The quick brown fox jumps over the lazy dog»: n = 28, len = 43.
    const std::string s = "The quick brown fox jumps over the lazy dog";
    std::string alpha;
    for (char c : s)
        if (alpha.find(c) == std::string::npos) alpha += c;
    std::sort(alpha.begin(), alpha.end());
    KernelParams p;
    p.kernel = k;
    p.n = static_cast<uint32_t>(alpha.size());
    for (char c : s) p.target.push_back(static_cast<int32_t>(alpha.find(c)));
    return p;
}

struct Case {
    const char* name;
    KernelParams p;
    uint64_t begin, count;
};

std::vector<Case> cases() {
    std::vector<Case> out;
    out.push_back({"monkey-random", fox(KernelId::MonkeyRandom), (1ull << 40) + 7, 3000000});
    out.push_back({"monkey-random hi", fox(KernelId::MonkeyRandom), (1ull << 63) - 1000000, 2000000});
    out.push_back({"monkey-brute", fox(KernelId::MonkeyBrute), 0, 3000000});
    out.push_back({"monkey-brute hi", fox(KernelId::MonkeyBrute), ~0ull - 1500000, 1500000});
    {
        // Короткий эталон: полное совпадение внутри диапазона (first_full).
        KernelParams p;
        p.kernel = KernelId::MonkeyBrute;
        p.n = 10;
        p.target = {9, 8, 7, 6, 5, 4, 3, 2, 1, 0}; // код 9876543210 > 2^32
        out.push_back({"monkey-brute full match > 2^32", p, 9876543210ull - 500000, 1000000});
        KernelParams r;
        r.kernel = KernelId::MonkeyRandom;
        r.n = 3;
        r.target = {0, 1, 2, 0, 1, 2, 0, 1};
        out.push_back({"monkey-random full match", r, 0, 2000000});
    }
    {
        KernelParams p;
        p.kernel = KernelId::Bbp;
        p.bbp.pos0 = 30000;
        p.bbp.positions = 4;
        p.bbp.block_terms = 1024;
        p.bbp.blocks = (30000 + 3 + 1024) / 1024;
        out.push_back({"bbp", p, 0, bbp_units(p.bbp)});
        KernelParams w;
        w.kernel = KernelId::Bbp;
        w.bbp = bbp_window(2);
        // Последние блоки окна у 10^6: большие знаменатели и показатели.
        out.push_back({"bbp near 1e6", w, bbp_units(w.bbp) - 64, 64});
    }
    {
        KernelParams p;
        p.kernel = KernelId::MillerRabin;
        p.mr_base = (1ull << 62) + 1;
        p.mr_step = 2;
        out.push_back({"miller-rabin 2^62", p, 0, 300000});
        KernelParams s;
        s.kernel = KernelId::MillerRabin;
        s.mr_base = 0;
        s.mr_step = 1;
        out.push_back({"miller-rabin small n", s, 0, 200000});
        KernelParams t;
        t.kernel = KernelId::MillerRabin;
        t.mr_base = ~0ull - 400000;
        t.mr_step = 1;
        out.push_back({"miller-rabin near 2^64", t, 0, 400000});
    }
    return out;
}

// Прогон диапазона на устройстве кусками заданного размера.
RangeResult gpu_range(GpuDevice& dev, uint64_t begin, uint64_t count, uint64_t piece) {
    RangeResult total;
    for (uint64_t b = begin; b < begin + count;) {
        const uint64_t c = std::min(piece, begin + count - b);
        total.merge(dev.run_range(b, c));
        b += c;
    }
    return total;
}

void check_device(GpuDevice& dev) {
    INFO("device " << dev.info().name << " (" << dev.info().api << ", u64 " << dev.info().u64_mode << ")");
    for (const Case& c : cases()) {
        INFO("case " << c.name);
        std::vector<uint64_t> cpu_acc(c.p.bbp.positions, 0), gpu_acc;
        const RangeResult cpu = cpu_run_range(c.p, c.begin, c.count, cpu_acc.data());
        REQUIRE(cpu.hits > 0);
        dev.prepare(c.p);
        for (uint64_t piece : {c.count, c.count / 3 + 1, uint64_t(4099)}) {
            INFO("piece " << piece);
            dev.reset_bbp_acc();
            const RangeResult gpu = gpu_range(dev, c.begin, c.count, piece);
            CHECK(gpu.count == cpu.count);
            CHECK(gpu.hits == cpu.hits);
            CHECK(gpu.checksum == cpu.checksum);
            CHECK(gpu.first_full == cpu.first_full);
            if (c.p.kernel == KernelId::Bbp) {
                dev.read_bbp_acc(gpu_acc);
                CHECK(gpu_acc == cpu_acc);
            }
        }
    }
}

} // namespace

TEST_CASE("Vulkan matches the CPU reference for every kernel", "[parity][vulkan]") {
    if (!vulkan_compiled()) {
        WARN("built without Vulkan — skipping");
        return;
    }
    for (bool emulated : {false, true}) {
        VulkanOptions opt;
        opt.allow_cpu_device = true;
        opt.force_emulated_u64 = emulated;
        std::string err;
        auto dev = create_vulkan_device(opt, &err);
        if (!dev) {
            // В CI с lavapipe устройство обязано быть: тихий пропуск скрыл бы,
            // что паритет на GPU не проверялся.
            const char* req = std::getenv("MONKEY_REQUIRE_VULKAN");
            if (req && *req == '1') FAIL("Vulkan required but unavailable: " << err);
            WARN("no Vulkan device (" << err << ") — skipping");
            return;
        }
        if (!emulated && dev->info().u64_mode != "native") {
            WARN("device has no shaderInt64 — native variant not tested");
            continue;
        }
        check_device(*dev);
    }
}

TEST_CASE("CUDA matches the CPU reference for every kernel", "[parity][cuda]") {
    if (!cuda_compiled()) {
        WARN("built without CUDA — skipping");
        return;
    }
    std::string err;
    auto dev = create_cuda_device(CudaOptions{}, &err);
    if (!dev) {
        WARN("no CUDA device (" << err << ") — skipping");
        return;
    }
    check_device(*dev);
}
