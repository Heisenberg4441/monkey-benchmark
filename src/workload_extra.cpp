#include "workload_extra.h"

#include "pi_digits.h"
#include "workloads/bbp_kernel.h"
#include "workloads/miller_rabin_kernel.h"

#include <cstdint>

namespace monkey {

// ---------------------------------------------------------------------------
// BBPPiWorkload
// ---------------------------------------------------------------------------

kernel::BbpParams BBPPiWorkload::default_window() {
    kernel::BbpParams p{};
    p.positions = 1024;
    p.pos0 = pi_hex_count() - 8 - p.positions; // 8 цифр последней позиции есть в файле
    p.block_terms = 4096;
    const uint64_t pos_max = p.pos0 + p.positions - 1;
    p.blocks = static_cast<uint32_t>((pos_max + p.block_terms) / p.block_terms);
    return p;
}

void BBPPiWorkload::prepare(const Config&) {
    p_ = default_window();
    units_ = static_cast<uint64_t>(p_.positions) * p_.blocks;
}

void BBPPiWorkload::execute_batch(Counter counter_start, uint64_t batch_size,
                                  WorkloadResult& out) const {
    const uint64_t start = counter_lo(counter_start);
    for (uint64_t i = 0; i < batch_size; ++i) {
        const uint64_t u = (start + i) % units_;
        out.checksum += kernel::contrib(u, kernel::bbp_unit(p_, u));
    }
    out.iterations += batch_size;
}

bool BBPPiWorkload::verify() const {
    // Полные суммы позиций 0..63 (все блоки + хвост) против встроенного π,
    // по 8 цифр на позицию.
    for (uint64_t d = 0; d < 64; ++d) {
        uint64_t s = kernel::bbp_tail(d);
        for (uint64_t k0 = 0; k0 <= d; k0 += p_.block_terms) s += kernel::bbp_terms(d, k0, k0 + p_.block_terms);
        if (kernel::bbp_digits(s) != pi_hex_word(d)) return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// MillerRabinWorkload
// ---------------------------------------------------------------------------

void MillerRabinWorkload::prepare(const Config& cfg) { seed_ = cfg.seed; }

void MillerRabinWorkload::execute_batch(Counter counter_start, uint64_t batch_size,
                                        WorkloadResult& out) const {
    const uint64_t start = counter_lo(counter_start);
    uint64_t done = 0;
    for (uint64_t i = 0; i < batch_size; ++i) {
        const uint64_t n = start + i;
        ++done; // каждый counter — одна операция, включая n < 2
        if (kernel::miller_rabin_is_prime(n)) {
            out.checksum += kernel::contrib(n, 1); // order-independent Σ mod 2^64
        }
    }
    out.iterations += done;
}

bool MillerRabinWorkload::verify() const {
    // Проверяем известные простые и составные.
    struct {
        uint64_t n;
        bool is_prime;
    } cases[] = {
        {2, true},     {3, true},     {5, true},     {7, true},
        {11, true},    {13, true},    {17, true},    {19, true},
        {0, false},    {1, false},    {4, false},    {6, false},
        {9, false},    {15, false},   {21, false},   {25, false},
        {100, false},  {102, false},  {313, true},   {997, true},
        {1000000007ULL, true},  // 1e9+7 — известное простое
        {1000000008ULL, false},
        {1099511627689ULL, true}, // простое > 10^12
        {1099511627695ULL, false},
        {3314192745739ULL, true}, // детерминированные свидетели до 3.3e14
        {3314192745741ULL, false},
    };

    for (const auto& c : cases) {
        if (c.n == 0) {
            // n=0 специально — is_prime(0) должен быть false
            if (kernel::miller_rabin_is_prime(0)) return false;
            continue;
        }
        if (c.n == 1) {
            if (kernel::miller_rabin_is_prime(1)) return false;
            continue;
        }
        const bool r = kernel::miller_rabin_is_prime(c.n);
        if (r != c.is_prime) return false;
    }
    // Детерминированность: повторный вызов — тот же результат.
    if (kernel::miller_rabin_is_prime(313) != kernel::miller_rabin_is_prime(313))
        return false;
    return true;
}

} // namespace monkey
