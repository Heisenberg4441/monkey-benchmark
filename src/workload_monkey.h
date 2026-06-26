#pragma once

#include <cstdint>
#include <vector>

#include "workload.h"

namespace monkey {

// Текущая логика бенчмарка под интерфейсом IWorkload: ищет counter, чей кандидат
// совпадает с эталоном (random — кандидат из PRNG, brute — counter как число по
// основанию n). Per-counter математика — в workloads/monkey_kernel.h.
class MonkeyWorkload : public IWorkload {
public:
    const char* name() const override { return "monkey"; }
    void prepare(const Config& cfg) override;
    void execute_batch(uint64_t counter_start, uint64_t batch_size,
                       WorkloadResult& out) const override;
    bool verify() const override;

private:
    int mode_ = 0; // 0 = random, 1 = brute
    uint32_t seed_ = 0;
    int n_ = 0;
    int len_ = 0;
    std::vector<int> target_;
};

} // namespace monkey
