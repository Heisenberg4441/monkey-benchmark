#include "workload_extra.h"

#include "workloads/bbp_kernel.h"
#include "workloads/miller_rabin_kernel.h"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <vector>

namespace monkey {

// ---------------------------------------------------------------------------
// BBPPiWorkload
// ---------------------------------------------------------------------------

void BBPPiWorkload::prepare(const Config& cfg) {
    seed_ = cfg.seed;

    // Загрузка эталонных hex-знаков π.
    std::ifstream f("reference/pi_hex.txt");
    if (f.is_open()) {
        std::string content((std::istreambuf_iterator<char>(f)),
                             std::istreambuf_iterator<char>());
        for (char c : content) {
            if (c >= '0' && c <= '9')
                pi_digits_.push_back(static_cast<uint8_t>(c - '0'));
            else if (c >= 'a' && c <= 'f')
                pi_digits_.push_back(static_cast<uint8_t>(c - 'a' + 10));
            else if (c >= 'A' && c <= 'F')
                pi_digits_.push_back(static_cast<uint8_t>(c - 'A' + 10));
        }
    }
}

void BBPPiWorkload::execute_batch(Counter counter_start, uint64_t batch_size,
                                  WorkloadResult& out) const {
    const uint64_t start = counter_lo(counter_start);
    uint64_t done = 0;
    for (uint64_t i = 0; i < batch_size; ++i) {
        const uint64_t d = start + i;
        ++done;
        const int digit = kernel::bbp_hex_digit(d);
        out.checksum ^= static_cast<uint64_t>(digit) << (static_cast<uint64_t>(i) & 63u);
    }
    out.iterations += done;
}

bool BBPPiWorkload::verify() const {
    if (pi_digits_.empty()) return false;

    // Сверяем первые 1024 знака.
    constexpr uint64_t kCheck = 1024;
    const uint64_t limit = (pi_digits_.size() < kCheck) ? pi_digits_.size() : kCheck;
    for (uint64_t d = 0; d < limit; ++d) {
        if (kernel::bbp_hex_digit(d) != static_cast<int>(pi_digits_[d])) {
            return false;
        }
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
        if (n < 2) continue;
        ++done;
        if (n == 2 || n == 3 || kernel::miller_rabin_is_prime(n)) {
            out.checksum ^= n;
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
