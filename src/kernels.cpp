#include "kernels.h"

#include "pi_digits.h"
#include "workloads/arith.h"
#include "workloads/miller_rabin_kernel.h"
#include "workloads/monkey_kernel.h"

namespace monkey {

const char* kernel_name(KernelId k) {
    switch (k) {
    case KernelId::MonkeyRandom:
        return "monkey-random";
    case KernelId::MonkeyBrute:
        return "monkey-brute";
    case KernelId::Bbp:
        return "bbp";
    case KernelId::MillerRabin:
        return "miller-rabin";
    }
    return "unknown";
}

kernel::BbpParams bbp_window(uint32_t positions, uint32_t block_terms) {
    kernel::BbpParams p{};
    p.positions = positions;
    p.pos0 = pi_hex_count() - 8 - positions; // 8 цифр последней позиции есть в файле
    p.block_terms = block_terms;
    const uint64_t pos_max = p.pos0 + positions - 1;
    p.blocks = static_cast<uint32_t>((pos_max + block_terms) / block_terms);
    return p;
}

uint64_t bbp_units(const kernel::BbpParams& p) {
    return static_cast<uint64_t>(p.positions) * p.blocks;
}

namespace {

// Горячие циклы — отдельная функция на kernel: без switch на индекс.
RangeResult run_monkey_random(const KernelParams& p, uint64_t begin, uint64_t count) {
    RangeResult res;
    const int32_t* target = p.target.data();
    const uint32_t len = p.len(), n = p.n, seed = p.seed;
    uint64_t hits = 0, sum = 0, first = UINT64_MAX;
    for (uint64_t i = begin; i < begin + count; ++i) {
        const uint32_t r = kernel::monkey_random_prefix(seed, i, target, len, n);
        if (r != 0) {
            ++hits;
            sum += kernel::contrib(i, r);
            if (r == len && i < first) first = i;
        }
    }
    res.count = count;
    res.hits = hits;
    res.checksum = sum;
    res.first_full = first;
    return res;
}

RangeResult run_monkey_brute(const KernelParams& p, uint64_t begin, uint64_t count) {
    RangeResult res;
    const int32_t* target = p.target.data();
    const uint32_t len = p.len(), n = p.n;
    uint64_t hits = 0, sum = 0, first = UINT64_MAX;
    for (uint64_t i = begin; i < begin + count; ++i) {
        const uint32_t r = kernel::monkey_brute_suffix(i, target, len, n);
        if (r != 0) {
            ++hits;
            sum += kernel::contrib(i, r);
            if (r == len && i < first) first = i;
        }
    }
    res.count = count;
    res.hits = hits;
    res.checksum = sum;
    res.first_full = first;
    return res;
}

RangeResult run_bbp(const KernelParams& p, uint64_t begin, uint64_t count, uint64_t* acc) {
    RangeResult res;
    for (uint64_t u = begin; u < begin + count; ++u) {
        const uint64_t r = kernel::bbp_unit(p.bbp, u);
        if (acc) acc[kernel::bbp_unit_position(p.bbp, u)] += r;
        if (r != 0) {
            ++res.hits;
            res.checksum += kernel::contrib(u, r);
        }
    }
    res.count = count;
    return res;
}

RangeResult run_miller_rabin(const KernelParams& p, uint64_t begin, uint64_t count) {
    RangeResult res;
    uint64_t hits = 0, sum = 0;
    for (uint64_t i = begin; i < begin + count; ++i) {
        if (kernel::miller_rabin_is_prime(p.mr_base + p.mr_step * i)) {
            ++hits;
            sum += kernel::contrib(i, 1);
        }
    }
    res.count = count;
    res.hits = hits;
    res.checksum = sum;
    return res;
}

} // namespace

RangeResult cpu_run_range(const KernelParams& p, uint64_t begin, uint64_t count,
                          uint64_t* bbp_acc) {
    switch (p.kernel) {
    case KernelId::MonkeyRandom:
        return run_monkey_random(p, begin, count);
    case KernelId::MonkeyBrute:
        return run_monkey_brute(p, begin, count);
    case KernelId::Bbp:
        return run_bbp(p, begin, count, bbp_acc);
    case KernelId::MillerRabin:
        return run_miller_rabin(p, begin, count);
    }
    return {};
}

namespace {

bool fail(std::string* why, const char* msg) {
    if (why) *why = msg;
    return false;
}

} // namespace

bool kernel_selfcheck(const KernelParams& p, std::string* why) {
    switch (p.kernel) {
    case KernelId::MonkeyRandom:
    case KernelId::MonkeyBrute: {
        if (p.n < 2 || p.target.empty()) return fail(why, "alphabet needs >= 2 symbols");
        for (int32_t t : p.target)
            if (t < 0 || static_cast<uint32_t>(t) >= p.n) return fail(why, "target index out of range");
        // Канон маппинга (зафиксирован в test_kernels_monkey): seed 0x9e3779b9, n = 27.
        const uint32_t kat[8] = {26, 2, 13, 25, 26, 7, 1, 24};
        for (uint32_t pos = 0; pos < 8; ++pos)
            if (kernel::monkey_char(0x9e3779b9u, 0, pos, 27) != kat[pos])
                return fail(why, "Philox candidate mapping mismatch");
        if (p.kernel == KernelId::MonkeyBrute) {
            // Индекс, кодирующий эталон (если влезает в 64 бита), обязан совпасть.
            uint64_t code = 0;
            bool fits = true;
            for (int32_t t : p.target) {
                if (code > (UINT64_MAX - static_cast<uint64_t>(t)) / p.n) { fits = false; break; }
                code = code * p.n + static_cast<uint64_t>(t);
            }
            if (fits && kernel::monkey_brute_suffix(code, p.target.data(), p.len(), p.n) != p.len())
                return fail(why, "brute encoding mismatch");
        }
        return true;
    }
    case KernelId::Bbp:
        // Полные суммы позиций 0..63 против встроенного π, по 8 цифр.
        for (uint64_t d = 0; d < 64; ++d) {
            uint64_t s = kernel::bbp_tail(d);
            for (uint64_t k0 = 0; k0 <= d; k0 += 16) s += kernel::bbp_terms(d, k0, k0 + 16);
            if (kernel::bbp_digits(s) != pi_hex_word(d)) return fail(why, "BBP digits mismatch");
        }
        if (p.bbp.blocks == 0 || p.bbp.block_terms == 0 || p.bbp.positions == 0)
            return fail(why, "empty BBP window");
        if (p.bbp.pos0 + p.bbp.positions + 8 > pi_hex_count())
            return fail(why, "BBP window beyond embedded pi digits");
        return true;
    case KernelId::MillerRabin: {
        const uint64_t primes[] = {2, 3, 5, 997, 1000000007ull, 18446744073709551557ull};
        const uint64_t comps[] = {0, 1, 4, 4033, 4681, 3825123056546413051ull};
        for (uint64_t x : primes)
            if (!kernel::miller_rabin_is_prime(x)) return fail(why, "Miller-Rabin rejects a prime");
        for (uint64_t x : comps)
            if (kernel::miller_rabin_is_prime(x)) return fail(why, "Miller-Rabin accepts a composite");
        return true;
    }
    }
    return fail(why, "unknown kernel");
}

} // namespace monkey
