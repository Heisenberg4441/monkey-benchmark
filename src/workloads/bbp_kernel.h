#pragma once

#include <cstdint>

#include "arith.h"

// BBP kernel: hex-цифры π по формуле Bailey-Borwein-Plouffe (1996)
//   frac(16^d·π) = frac(4·S1 - 2·S4 - S5 - S6),
//   Sj = Σ_{k<=d} (16^(d-k) mod (8k+j)) / (8k+j) + Σ_{k>d} 16^(d-k) / (8k+j).
// Верхние биты frac(16^d·π) — hex-цифры π начиная с позиции d (0 — первая
// после «3.»).
//
// Целочисленный fixed-point вместо double: дробь mod 1 хранится как uint64
// (единица младшего разряда 2^-64), сложение — с естественным переполнением.
// Член ряда — floor(2^64·x/m) при x < m < 2^32: два шага деления столбиком на
// 32-битных лимбах. 16^e mod m — Montgomery с R = 2^32 (arith.h). Результат
// бит-в-бит одинаков на CPU, CUDA и GLSL (src/shaders/kernels.comp) и не
// требует FP64 (на Apple GPU его нет). Ограничение: 8d+6 < 2^32, т.е.
// d < 5.3·10^8.
//
// Чётные знаменатели сокращаются до нечётного модуля, дробь та же самая:
//   16^e mod (8k+4) / (8k+4) = (4·16^(e-1) mod (2k+1)) / (2k+1),
//   16^e mod (8k+6) / (8k+6) = (8·16^(e-1) mod (4k+3)) / (4k+3),   e >= 1.
//
// Погрешность: floor теряет < 1 ulp на член; веса 4-2-1-1 в сумме дают ноль,
// поэтому систематический сдвиг сокращается и остаётся случайное блуждание
// ~sqrt(22·d/12) ulp (~2^10 при d ~ 10^6). Верхние 32 бита (8 цифр) верны,
// если позиция не лежит у границы округления; окна suite проверяются целиком
// против reference/pi_hex.txt.

namespace monkey {
namespace kernel {

// Единица работы — блок из K членов одной позиции: индекс u -> позиция
// pos0 + u / blocks, k в [b·K, b·K + K), b = u % blocks. Так позиция у 10^6
// (~4·10^6 членов, доли секунды на ядро) режется на куски короче чанка, а
// частичные суммы складываются по позициям mod 2^64 в любом порядке.
struct BbpParams {
    uint64_t pos0;        // первая позиция окна
    uint32_t positions;   // число позиций
    uint32_t block_terms; // K — членов ряда на единицу работы
    uint32_t blocks;      // блоков на позицию: ceil((pos_max + 1) / K)
};

KERNEL_HD inline uint32_t addmod32(uint32_t a, uint32_t b, uint32_t m) {
    return a >= m - b ? a - (m - b) : a + b;
}

// floor(2^64·x/m) для x < m < 2^32: частное по двум 32-битным лимбам.
KERNEL_HD inline uint64_t bbp_frac(uint32_t x, uint32_t m) {
    const uint64_t a = static_cast<uint64_t>(x) << 32;
    const uint64_t q1 = a / m;
    const uint64_t r1 = a - q1 * m;
    const uint64_t q0 = (r1 << 32) / m;
    return (q1 << 32) | q0;
}

// Вклад членов k в [k0, k1) ∩ [0, d] позиции d: Σ (4·T1 - 2·T4 - T5 - T6) mod 2^64.
// Четыре модуля возводятся в одну и ту же степень 16^(d-k-1) одновременно —
// независимые цепочки умножений, которые CPU и GPU исполняют параллельно (ILP).
KERNEL_HD inline uint64_t bbp_terms(uint64_t d, uint64_t k0, uint64_t k1) {
    uint64_t sum = 0;
    const uint64_t kend = k1 < d ? k1 : d; // k < d: e = d - k >= 1
    for (uint64_t k = k0; k < kend; ++k) {
        const uint32_t m[4] = {static_cast<uint32_t>(8 * k + 1), static_cast<uint32_t>(2 * k + 1),
                               static_cast<uint32_t>(8 * k + 5), static_cast<uint32_t>(4 * k + 3)};
        const uint32_t c[4] = {16, 4, 16, 8}; // множитель перед 16^(e-1), см. шапку
        uint32_t minv[4], r[4], r16[4];
        for (int j = 0; j < 4; ++j) {
            minv[j] = inv32(m[j]);
            r[j] = (0u - m[j]) % m[j]; // 1 в форме Монтгомери: 2^32 mod m (0 при m = 1)
            r16[j] = r[j];             // 16 в форме Монтгомери — четыре удвоения единицы
            for (int q = 0; q < 4; ++q) r16[j] = addmod32(r16[j], r16[j], m[j]);
        }
        // 16^f слева направо: квадрат и, на единичных битах, умножение на 16.
        // (Вариант «умножение на 16 = 4 удвоения» на M4 Pro медленнее на 8%,
        // аппаратный % быстрее на 5%, но хуже на x86 и GPU — см. BENCH.md.)
        const uint64_t f = d - k - 1;
        if (f != 0) {
            for (int bit = 63 - clz64(f); bit >= 0; --bit) {
                for (int j = 0; j < 4; ++j) r[j] = mont32_mul(r[j], r[j], m[j], minv[j]);
                if ((f >> bit) & 1u) {
                    for (int j = 0; j < 4; ++j) r[j] = mont32_mul(r[j], r16[j], m[j], minv[j]);
                }
            }
        }
        // REDC с множителем c: выход из формы Монтгомери и умножение за один шаг.
        uint64_t t[4];
        for (int j = 0; j < 4; ++j) t[j] = bbp_frac(mont32_mul(r[j], c[j], m[j], minv[j]), m[j]);
        sum += 4 * t[0] - 2 * t[1] - t[2] - t[3];
    }
    if (k0 <= d && d < k1) { // k = d: 16^0 / (8d + j), знаменатель любой чётности
        const uint32_t den[4] = {static_cast<uint32_t>(8 * d + 1), static_cast<uint32_t>(8 * d + 4),
                                 static_cast<uint32_t>(8 * d + 5), static_cast<uint32_t>(8 * d + 6)};
        uint64_t t[4];
        for (int j = 0; j < 4; ++j) t[j] = bbp_frac(1u % den[j], den[j]);
        sum += 4 * t[0] - 2 * t[1] - t[2] - t[3];
    }
    return sum;
}

// Хвост k > d: 16^(d-k) / (8k+j) = 2^(64-4t) / (8k+j) в единицах 2^-64,
// t = k - d; при t >= 16 член меньше ulp.
KERNEL_HD inline uint64_t bbp_tail(uint64_t d) {
    uint64_t sum = 0;
    for (uint32_t t = 1; t < 16; ++t) {
        const uint64_t num = 1ull << (64 - 4 * t);
        const uint64_t k = d + t;
        const uint64_t t1 = num / (8 * k + 1), t4 = num / (8 * k + 4);
        const uint64_t t5 = num / (8 * k + 5), t6 = num / (8 * k + 6);
        sum += 4 * t1 - 2 * t4 - t5 - t6;
    }
    return sum;
}

// Индекс позиции (0..positions-1) единицы работы u.
KERNEL_HD inline uint32_t bbp_unit_position(const BbpParams& p, uint64_t u) {
    return static_cast<uint32_t>(u / p.blocks);
}

// Частичная сумма единицы работы u (без хвоста).
KERNEL_HD inline uint64_t bbp_unit(const BbpParams& p, uint64_t u) {
    const uint64_t w = u / p.blocks;
    const uint64_t b = u - w * p.blocks;
    const uint64_t k0 = b * p.block_terms;
    return bbp_terms(p.pos0 + w, k0, k0 + p.block_terms);
}

// 8 hex-цифр π с позиции d из полной суммы позиции (все блоки + хвост).
KERNEL_HD inline uint32_t bbp_digits(uint64_t position_sum) {
    return static_cast<uint32_t>(position_sum >> 32);
}

} // namespace kernel
} // namespace monkey
