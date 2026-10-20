#pragma once

#include <cstdint>
#include <vector>

#include "workload.h"
#include "workloads/bbp_kernel.h"

namespace monkey {

// BBP workload: вычисляет hex-знаки π через формулу Bailey-Borwein-Plouffe
// (целочисленный fixed-point, workloads/bbp_kernel.h). Counter = единица работы
// «блок из K членов одной позиции окна» (по модулю размера окна).
// Тяжёлая арифметика (Montgomery-возведение + деление столбиком) делает этот
// workload вычислительно-насыщенным, в отличие от monkey (только PRNG + сравнение).
class BBPPiWorkload : public IWorkload {
public:
    const char* name() const override { return "bbp"; }
    void prepare(const Config& cfg) override;
    void execute_batch(Counter counter_start, uint64_t batch_size,
                       WorkloadResult& out) const override;
    bool verify() const override;

    // Окно по умолчанию: последние позиции встроенного pi_hex.txt.
    static kernel::BbpParams default_window();

private:
    kernel::BbpParams p_{};
    uint64_t units_ = 1; // positions · blocks
};

// Miller-Rabin workload: тест простоты 64-битных целых.
// Каждый counter = целое число n для проверки на простоту.
// Детерминирован для всех n < 2^64: свидетели Jim Sinclair
// {2, 325, 9375, 28178, 450775, 9780504, 1795265022}, каждый берётся mod n.
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
