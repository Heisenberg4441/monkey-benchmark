#include "workload_monkey.h"

#include "workloads/monkey_kernel.h"

#include <cstdint>

namespace monkey {

void MonkeyWorkload::prepare(const Config& cfg) {
    mode_ = (cfg.mode == Mode::Brute) ? 1 : 0;
    seed_ = cfg.seed;
    n_ = cfg.n;
    len_ = cfg.len;
    target_ = cfg.target_idx;
}

void MonkeyWorkload::execute_batch(uint64_t counter_start, uint64_t batch_size,
                                   WorkloadResult& out) const {
    // Локальные копии в регистрах: ничего не пишем сквозь out-ссылку в горячем
    // цикле (иначе компилятор перечитывает поля каждую итерацию).
    const int* target = target_.data();
    const int mode = mode_;
    const uint32_t seed = seed_;
    const int len = len_;
    const int n = n_;

    uint64_t done = 0;
    for (uint64_t i = 0; i < batch_size; ++i) {
        const uint64_t c = counter_start + i;
        ++done;
        if (kernel::monkey_match(mode, seed, c, target, len, n)) {
            out.solution_found = true;
            out.solution_counter = c;
            break;
        }
    }
    out.iterations += done;
}

bool MonkeyWorkload::verify() const {
    if (n_ <= 0 || len_ <= 0 || static_cast<int>(target_.size()) != len_) {
        return false;
    }

    // 1) monkey_char детерминирован и в диапазоне [0, n).
    for (uint32_t p = 0; p < static_cast<uint32_t>(len_); ++p) {
        for (uint64_t c = 0; c < 8; ++c) {
            const int a = kernel::monkey_char(seed_, c, p, n_);
            const int b = kernel::monkey_char(seed_, c, p, n_);
            if (a != b || a < 0 || a >= n_) {
                return false;
            }
        }
    }

    // 2) brute: counter, кодирующий эталон, обязан совпасть; соседний — нет.
    if (mode_ == 1) {
        uint64_t c = 0;
        bool encodable = true;
        const uint64_t base = static_cast<uint64_t>(n_);
        for (int p = 0; p < len_; ++p) {
            const uint64_t d = static_cast<uint64_t>(target_[p]);
            if (c > (UINT64_MAX - d) / base) { // переполнение uint64
                encodable = false;
                break;
            }
            c = c * base + d;
        }
        if (encodable) {
            if (!kernel::monkey_match(1, seed_, c, target_.data(), len_, n_)) {
                return false;
            }
            // Соседний counter меняет последний разряд (если он не максимум) —
            // совпадения быть не должно.
            if (target_[len_ - 1] != n_ - 1 &&
                kernel::monkey_match(1, seed_, c + 1, target_.data(), len_, n_)) {
                return false;
            }
        }
    }
    return true;
}

} // namespace monkey
