#pragma once

#include <cstdint>

#include "arith.h"

// Miller-Rabin deterministic primality test for 64-bit integers.
// Uses known witnesses for n < 2^64 per Jim Sinclair (2011):
//   n < 2^64: {2, 325, 9375, 28178, 450775, 9780504, 1795265022}
// Stateless: результат зависит только от n (counter'а), без PRNG.
//
// Арифметика — Montgomery с R = 2^64 (arith.h): ни одного 128-битного деления
// в горячем пути, одинаково на CPU, CUDA и GLSL (src/shaders/kernels.comp).

namespace monkey {
namespace kernel {

// Montgomery-контекст одного нечётного n: n^-1 mod 2^64 и представления 1 и -1.
struct MrCtx {
    uint64_t n;
    uint64_t ninv;
    uint64_t one;  // R mod n
    uint64_t mone; // (n - 1)·R mod n
};

KERNEL_HD inline uint64_t mr_mul(uint64_t a, uint64_t b, const MrCtx& c) {
    return mont64_mul(a, b, c.n, c.ninv);
}

// base^e для base в форме Монтгомери (справа налево).
KERNEL_HD inline uint64_t mr_pow(uint64_t base, uint64_t e, const MrCtx& c) {
    uint64_t r = c.one;
    while (e) {
        if (e & 1u) r = mr_mul(r, base, c);
        e >>= 1u;
        if (e) base = mr_mul(base, base, c);
    }
    return r;
}

// 2^e слева направо: умножение на основание 2 — это модульное удвоение, а не
// mont-умножение. Для составных обычно проверяется только свидетель 2, поэтому
// это основной путь workload'а.
KERNEL_HD inline uint64_t mr_pow2(uint64_t e, const MrCtx& c) {
    uint64_t r = c.one;
    for (int i = 63 - clz64(e); i >= 0; --i) {
        r = mr_mul(r, r, c);
        if ((e >> i) & 1u) r = addmod64(r, r, c.n);
    }
    return r;
}

// Сильная проверка по x = a^d (форма Монтгомери), n - 1 = d·2^s.
KERNEL_HD inline bool mr_strong(uint64_t x, int s, const MrCtx& c) {
    if (x == c.one || x == c.mone) return true;
    for (int r = 1; r < s; ++r) {
        x = mr_mul(x, x, c);
        if (x == c.mone) return true;
        if (x == c.one) return false;
    }
    return false;
}

// Детерминированный тест простоты для n < 2^64.
KERNEL_HD inline bool miller_rabin_is_prime(uint64_t n) {
    if (n < 2) return false;
    if (n < 4) return true;
    if ((n & 1u) == 0) return false;

    // n-1 = d * 2^s, d — нечётное.
    const int s = ctz64(n - 1);
    const uint64_t d = (n - 1) >> s;

    MrCtx c;
    c.n = n;
    c.ninv = inv64(n);
    c.one = pow2_64_mod(n);
    c.mone = n - c.one;

    // Свидетели — каждый берётся mod n, нулевой пропускается. Пропуск a >= n
    // (вместо редукции) давал ложные простые: 4033 и 4681 проходят по {2, 325}.
    if (!mr_strong(mr_pow2(d, c), s, c)) return false;

    // R^2 mod n — удвоением R mod n ещё 64 раза; нужен только тем n, что
    // прошли свидетеля 2 (простые и редкие псевдопростые).
    uint64_t r2 = c.one;
    for (int i = 0; i < 64; ++i) r2 = addmod64(r2, r2, n);

    const uint64_t witnesses[] = {325, 9375, 28178, 450775, 9780504, 1795265022};
    for (uint64_t a : witnesses) {
        a %= n;
        if (a == 0) continue;
        if (!mr_strong(mr_pow(mr_mul(a, r2, c), d, c), s, c)) return false;
    }
    return true;
}

// --- Пошаговый вариант для SIMT (CUDA, GLSL) -----------------------------------
//
// На GPU 32 потока SIMD-группы исполняют одну инструкцию: в miller_rabin_is_prime
// поток с составным числом (почти всегда 1 свидетель) ждёт соседа с простым
// (7 свидетелей), и ~78% групп содержат хотя бы одно простое. Автомат ниже
// разбивает тест на шаги «не больше одного mont-умножения»: поток, закончивший
// число, сразу берёт следующее, и в каждой итерации цикла все потоки группы
// заняты одним и тем же умножением. Результат бит-в-бит тот же, что у
// miller_rabin_is_prime (проверяет test_miller_rabin).

struct MrLane {
    MrCtx c;
    uint64_t d;    // n - 1 = d·2^s, d нечётное
    uint64_t x;    // аккумулятор в форме Монтгомери
    uint64_t base; // основание текущего свидетеля (w >= 1) в форме Монтгомери
    uint64_t r2;   // R^2 mod n, считается при переходе к свидетелю 1
    int s;
    int w;     // номер свидетеля: 0 — основание 2, 1..6 — mr_witness_base(w)
    int bit;   // фаза kMrExp: следующий бит показателя; kMrStrong: число квадратов
    int phase; // kMrExp | kMrMul | kMrStrong | kMrCheck | kMrInit
};

constexpr int kMrExp = 0;    // квадрат (+ удвоение для свидетеля 2)
constexpr int kMrMul = 1;    // умножение на основание
constexpr int kMrStrong = 2; // квадраты сильной проверки
constexpr int kMrCheck = 3;  // степень готова (d = 1): сильная проверка без умножения
constexpr int kMrInit = 4;   // основание a·R = mont(a, R^2) — через общий слот умножения

// Результат шага: 0 — продолжать, 1 — простое, 2 — составное.
constexpr int kMrContinue = 0;
constexpr int kMrPrime = 1;
constexpr int kMrComposite = 2;

KERNEL_HD inline uint64_t mr_witness_base(int w) {
    const uint64_t ws[6] = {325, 9375, 28178, 450775, 9780504, 1795265022};
    return ws[w - 1];
}

// Начало показателя: старший бит уже учтён в x. Без рекурсии (её нет в GLSL):
// при d = 1 сильная проверка — следующим шагом.
KERNEL_HD inline void mr_lane_exp_begin(MrLane& L) {
    L.bit = 62 - clz64(L.d);
    L.phase = L.bit < 0 ? kMrCheck : kMrExp;
}

// Начало проверки свидетеля w (>= 0) или итог, если свидетели кончились.
KERNEL_HD inline int mr_lane_witness(MrLane& L, int w) {
    for (; w < 7; ++w) {
        L.w = w;
        if (w == 0) {
            L.x = addmod64(L.c.one, L.c.one, L.c.n); // 2 в форме Монтгомери
            mr_lane_exp_begin(L);
            return kMrContinue;
        }
        if (w == 1) {
            L.r2 = L.c.one;
            for (int i = 0; i < 64; ++i) L.r2 = addmod64(L.r2, L.r2, L.c.n);
        }
        const uint64_t a = mr_witness_base(w) % L.c.n;
        if (a == 0) continue;
        L.x = a; // следующий шаг: x = mont(a, R^2) = a·R
        L.base = L.r2;
        L.phase = kMrInit;
        return kMrContinue;
    }
    return kMrPrime;
}

// Конец возведения в степень: сильная проверка x = a^d.
KERNEL_HD inline int mr_lane_strong_begin(MrLane& L) {
    if (L.x == L.c.one || L.x == L.c.mone) return mr_lane_witness(L, L.w + 1);
    L.bit = 1;
    if (L.bit >= L.s) return kMrComposite;
    L.phase = kMrStrong;
    return kMrContinue;
}

// Начало числа n; тривиальные n решаются сразу.
KERNEL_HD inline int mr_lane_start(MrLane& L, uint64_t n) {
    if (n < 2) return kMrComposite;
    if (n < 4) return kMrPrime;
    if ((n & 1u) == 0) return kMrComposite;
    L.s = ctz64(n - 1);
    L.d = (n - 1) >> L.s;
    L.c.n = n;
    L.c.ninv = inv64(n);
    L.c.one = pow2_64_mod(n);
    L.c.mone = n - L.c.one;
    return mr_lane_witness(L, 0);
}

// Один шаг — ровно одно mont-умножение вне ветвлений (на GPU его исполняют
// все потоки группы разом); фаза выбирает только второй множитель и дешёвую
// обработку результата.
KERNEL_HD inline int mr_lane_step(MrLane& L) {
    if (L.phase == kMrCheck) return mr_lane_strong_begin(L);
    const uint64_t y = (L.phase == kMrMul || L.phase == kMrInit) ? L.base : L.x;
    L.x = mr_mul(L.x, y, L.c);
    if (L.phase == kMrInit) {
        L.base = L.x;
        mr_lane_exp_begin(L);
        return kMrContinue;
    }
    if (L.phase == kMrStrong) {
        if (L.x == L.c.mone) return mr_lane_witness(L, L.w + 1);
        if (L.x == L.c.one) return kMrComposite;
        if (++L.bit >= L.s) return kMrComposite;
        return kMrContinue;
    }
    if (L.phase == kMrExp && ((L.d >> L.bit) & 1u)) {
        if (L.w != 0) {
            L.phase = kMrMul; // умножение на основание — следующим шагом
            return kMrContinue;
        }
        L.x = addmod64(L.x, L.x, L.c.n); // основание 2: удвоение
    }
    L.phase = kMrExp;
    return --L.bit < 0 ? mr_lane_strong_begin(L) : kMrContinue;
}

} // namespace kernel
} // namespace monkey
