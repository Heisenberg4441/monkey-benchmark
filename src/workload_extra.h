#pragma once

#include <cstdint>
#include <vector>

#include "workload.h"

namespace monkey {

// BBP workload: вычисляет hex-знаки π через формулу Bailey-Borwein-Plouffe.
// Каждый counter = позиция hex-знака (0 — первое после запятой).
// Тяжёлая арифметика (modular exponentiation + double-series) делает этот
// workload вычислительно-насыщенным, в отличие от monkey (только PRNG + memcmp).
class BBPPiWorkload : public IWorkload {
public:
    const char* name() const override { return "bbp"; }
    void prepare(const Config& cfg) override;
    void execute_batch(Counter counter_start, uint64_t batch_size,
                       WorkloadResult& out) const override;
    bool verify() const override;

private:
    uint32_t seed_ = 0;
    std::vector<uint8_t> pi_digits_; // эталонные hex-знаки (reference/pi_hex.txt)
};

// Miller-Rabin workload: тест простоты 64-битных целых.
// Каждый counter = целое число n для проверки на простоту.
// Детерминирован: использует известные свидетели для n < 3.3×10^14
// (witnesses: 2, 3, 5, 7, 11, 13, 17); для больших n — witnesses
// (2, 325, 9375, 28178, 450775, 9780504, 1795265022) по Jim Sinclair.
class MillerRabinWorkload : public IWorkload {
public:
    const char* name() const override { return "miller-rabin"; }
    void prepare(const Config& cfg) override;
    void execute_batch(Counter counter_start, uint64_t batch_size,
                       WorkloadResult& out) const override;
    bool verify() const override;

private:
    uint32_t seed_ = 0;
    // 64-битная арифметика: всё в регистрах/стеке, нет alloc.
};

} // namespace monkey
