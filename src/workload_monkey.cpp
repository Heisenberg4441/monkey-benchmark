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

    // Определяем, влезает ли всё пространство в 64 бита.
    if (mode_ == 1) {
        const uint64_t base = static_cast<uint64_t>(n_);
        Counter max_val = counter_from_u64(1);
        for (int i = 0; i < len_; ++i) {
            max_val = max_val * base;
        }
        max_val = max_val + counter_from_u64(static_cast<uint64_t>(-1)); // max_val -= 1
        // Если hi == 0, всё пространство в 64 битах.
        brute_fits_in_64_ = (counter_hi(max_val) == 0);
    } else {
        brute_fits_in_64_ = true;
    }
}

void MonkeyWorkload::execute_batch(Counter counter_start, uint64_t batch_size,
                                   WorkloadResult& out) const {
    const int* target = target_.data();
    const int mode = mode_;
    const uint32_t seed = seed_;
    const int len = len_;
    const int n = n_;

    uint64_t done = 0;

    if (mode == 1 && !brute_fits_in_64_) {
        // 128-битный brute: каждый counter — число в базе n.
        Counter c = counter_start;
        for (uint64_t i = 0; i < batch_size; ++i) {
            ++done;
            if (kernel::monkey_match_brute_128(c, target, len, n)) {
                out.solution_found = true;
                out.solution_counter = c;
                break;
            }
            c = c + counter_from_u64(1);
        }
    } else {
        // 64-битный путь (random всегда, brute если помещается).
        for (uint64_t i = 0; i < batch_size; ++i) {
            const uint64_t c64 = counter_lo(counter_start) + i;
            ++done;
            if (kernel::monkey_match(mode, seed, c64, target, len, n)) {
                out.solution_found = true;
                out.solution_counter = counter_from_u64(c64);
                break;
            }
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
        const uint64_t base = static_cast<uint64_t>(n_);
        // Пытаемся закодировать эталон в Counter (128 бит).
        Counter c = counter_from_u64(0);
        for (int p = 0; p < len_; ++p) {
            c = c * base;
            c = c + counter_from_u64(static_cast<uint64_t>(target_[p]));
        }

        if (brute_fits_in_64_) {
            const uint64_t c64 = counter_lo(c);
            if (!kernel::monkey_match(1, seed_, c64, target_.data(), len_, n_)) {
                return false;
            }
            if (target_[len_ - 1] != n_ - 1 &&
                kernel::monkey_match(1, seed_, c64 + 1, target_.data(), len_, n_)) {
                return false;
            }
        } else {
            if (!kernel::monkey_match_brute_128(c, target_.data(), len_, n_)) {
                return false;
            }
            if (target_[len_ - 1] != n_ - 1) {
                const Counter next = c + counter_from_u64(1);
                if (kernel::monkey_match_brute_128(next, target_.data(), len_, n_)) {
                    return false;
                }
            }
        }
    }
    return true;
}

} // namespace monkey
