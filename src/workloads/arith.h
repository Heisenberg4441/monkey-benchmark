#pragma once

#include <cstdint>

#if defined(_MSC_VER) && !defined(__CUDACC__)
#include <intrin.h>
#endif

// Общая целочисленная арифметика ядер: перемешивание для checksum, старшая
// половина 64×64 и Montgomery-умножение (64- и 32-битное). Header-only и
// __host__ __device__, как остальные ядра; GLSL-порт — src/shaders/arith.glsl.
//
// Все функции детерминированы и дают одинаковый результат на любой платформе;
// различается только реализация примитива mulhi64:
//   GCC/Clang хост — unsigned __int128;  MSVC x64 — _umul128;  MSVC ARM64 —
//   __umulh;  CUDA device — __umul64hi;  иначе — 32-битные лимбы.
// MONKEY_PORTABLE_ARITH принудительно включает переносимые 32-битные пути на
// любом компиляторе: так CI на clang/gcc проверяет код, который реально
// исполняется на MSVC и в GLSL.

#if defined(__CUDACC__)
#define KERNEL_HD __host__ __device__
#else
#define KERNEL_HD
#endif

#if !defined(MONKEY_PORTABLE_ARITH) && defined(__SIZEOF_INT128__) && !defined(__CUDACC__)
#define MONKEY_ARITH_INT128 1
#endif
// Интринсики MSVC — только вне nvcc: в host-проходе nvcc <intrin.h> не подключён.
#if !defined(MONKEY_PORTABLE_ARITH) && defined(_MSC_VER) && !defined(__CUDACC__)
#define MONKEY_ARITH_MSVC 1
#endif

namespace monkey {
namespace kernel {

// --- Перемешивание (финализатор splitmix64) ----------------------------------

KERNEL_HD inline uint64_t fmix64(uint64_t z) {
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
}

// Вклад индекса i с результатом r в order-independent checksum
// Σ contrib(i, r) mod 2^64. «Пустой» результат (r == 0) вклада не даёт: у
// monkey и MR это >95% индексов, и пропуск двух fmix64 экономит заметную долю
// времени. Пропуск работы ловится отдельно — по count и hits.
KERNEL_HD inline uint64_t contrib(uint64_t i, uint64_t r) {
    return r == 0 ? 0 : fmix64(fmix64(i) ^ r);
}

// --- Подсчёт нулевых битов (x != 0) -----------------------------------------------

KERNEL_HD inline int clz64(uint64_t x) {
#if defined(__CUDA_ARCH__)
    return __clzll(static_cast<long long>(x));
#elif defined(__GNUC__) || defined(__clang__)
    return __builtin_clzll(x);
#elif defined(MONKEY_ARITH_MSVC)
    unsigned long idx = 0;
    _BitScanReverse64(&idx, x);
    return 63 - static_cast<int>(idx);
#else
    int n = 0;
    while (!(x & (1ull << 63))) { x <<= 1; ++n; }
    return n;
#endif
}

KERNEL_HD inline int ctz64(uint64_t x) {
#if defined(__CUDA_ARCH__)
    return __ffsll(static_cast<long long>(x)) - 1;
#elif defined(__GNUC__) || defined(__clang__)
    return __builtin_ctzll(x);
#elif defined(MONKEY_ARITH_MSVC)
    unsigned long idx = 0;
    _BitScanForward64(&idx, x);
    return static_cast<int>(idx);
#else
    int n = 0;
    while (!(x & 1)) { x >>= 1; ++n; }
    return n;
#endif
}

// --- Старшая половина произведений ---------------------------------------------

KERNEL_HD inline uint32_t mulhi32(uint32_t a, uint32_t b) {
    return static_cast<uint32_t>((static_cast<uint64_t>(a) * b) >> 32);
}

// Переносимая 64×64 -> старшие 64 бита на 32-битных лимбах. Ровно этот
// алгоритм повторяет GLSL-порт (u64_mulhi в arith.glsl).
KERNEL_HD inline uint64_t mulhi64_portable(uint64_t a, uint64_t b) {
    const uint64_t a0 = static_cast<uint32_t>(a), a1 = a >> 32;
    const uint64_t b0 = static_cast<uint32_t>(b), b1 = b >> 32;
    const uint64_t p00 = a0 * b0, p01 = a0 * b1, p10 = a1 * b0, p11 = a1 * b1;
    const uint64_t mid = (p00 >> 32) + static_cast<uint32_t>(p01) + static_cast<uint32_t>(p10);
    return p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32);
}

KERNEL_HD inline uint64_t mulhi64(uint64_t a, uint64_t b) {
#if defined(__CUDA_ARCH__)
    return __umul64hi(a, b);
#elif defined(MONKEY_ARITH_INT128)
    return static_cast<uint64_t>((static_cast<unsigned __int128>(a) * b) >> 64);
#elif defined(MONKEY_ARITH_MSVC) && defined(_M_X64)
    unsigned __int64 hi = 0;
    (void)_umul128(a, b, &hi);
    return hi;
#elif defined(MONKEY_ARITH_MSVC) && defined(_M_ARM64)
    return __umulh(a, b);
#else
    return mulhi64_portable(a, b);
#endif
}

// --- Montgomery, R = 2^64 (нечётный модуль n < 2^64) --------------------------

// n^-1 mod 2^64 методом Ньютона: (3n)^2 верно в 5 младших битах, каждая
// итерация удваивает их число: 5 -> 10 -> 20 -> 40 -> 80.
KERNEL_HD inline uint64_t inv64(uint64_t n) {
    uint64_t x = (3 * n) ^ 2;
    for (int i = 0; i < 4; ++i) x *= 2 - n * x;
    return x;
}

// REDC через вычитание: t = a·b, m = lo(t)·n^-1, результат = hi(t) - hi(m·n)
// (mod n). Младшие половины t и m·n совпадают по построению, поэтому переноса
// нет и переполнения 128 бит тоже — работает для любого нечётного n < 2^64.
// Требует a·b < n·2^64 (достаточно a, b < n).
KERNEL_HD inline uint64_t mont64_mul(uint64_t a, uint64_t b, uint64_t n, uint64_t ninv) {
    const uint64_t lo = a * b;
    const uint64_t hi = mulhi64(a, b);
    const uint64_t mh = mulhi64(lo * ninv, n);
    const uint64_t r = hi - mh;
    return hi < mh ? r + n : r;
}

// 2^64 mod n для n > 0 без 128-битного деления. CPU берёт аппаратный остаток;
// переносимый путь — двоичное деление столбиком с clz(n) шагами (для n около
// 2^62 это 1 шаг). Результаты совпадают бит-в-бит.
KERNEL_HD inline uint64_t pow2_64_mod(uint64_t n) {
#if !defined(__CUDA_ARCH__) && !defined(MONKEY_PORTABLE_ARITH)
    return (0 - n) % n;
#else
    int s = 0;
    while ((n << s) < (1ull << 63)) ++s;
    uint64_t x = 0 - (n << s); // 2^64 - n·2^s, лежит в (0, 2^63]
    for (int i = s - 1; i >= 0; --i) {
        const uint64_t t = n << i;
        if (x >= t) x -= t;
    }
    return x >= n ? x - n : x;
#endif
}

// (a + b) mod n для a, b < n без переполнения.
KERNEL_HD inline uint64_t addmod64(uint64_t a, uint64_t b, uint64_t n) {
    return a >= n - b ? a - (n - b) : a + b;
}

// --- Montgomery, R = 2^32 (нечётный модуль m < 2^32) --------------------------

KERNEL_HD inline uint32_t inv32(uint32_t m) {
    uint32_t x = (3 * m) ^ 2;
    for (int i = 0; i < 3; ++i) x *= 2 - m * x;
    return x;
}

// Тот же REDC через вычитание на 32-битных словах (одно 32×32 -> 64 и одно
// mulhi32). Требует a·b < m·2^32.
KERNEL_HD inline uint32_t mont32_mul(uint32_t a, uint32_t b, uint32_t m, uint32_t minv) {
    const uint64_t t = static_cast<uint64_t>(a) * b;
    const uint32_t th = static_cast<uint32_t>(t >> 32);
    const uint32_t mh = mulhi32(static_cast<uint32_t>(t) * minv, m);
    const uint32_t r = th - mh;
    return th < mh ? r + m : r;
}

} // namespace kernel
} // namespace monkey
