// 64-битные целые для GLSL-ядер в двух вариантах SPIR-V:
//   U64_NATIVE — uint64_t (GL_EXT_shader_explicit_arithmetic_types_int64,
//                фича shaderInt64);
//   иначе      — эмуляция на uvec2 (x — младшие 32 бита, y — старшие).
// Ядра пишутся только через эти функции, поэтому оба варианта исполняют один и
// тот же алгоритм; тест паритета гоняет оба на реальном устройстве.
#ifndef MONKEY_U64_GLSL
#define MONKEY_U64_GLSL

#ifdef U64_NATIVE

#define U64 uint64_t
U64 u64(uint lo, uint hi) { return (uint64_t(hi) << 32) | uint64_t(lo); }
uint u64_lo(U64 a) { return uint(a); }
uint u64_hi(U64 a) { return uint(a >> 32); }
U64 u64_add(U64 a, U64 b) { return a + b; }
U64 u64_sub(U64 a, U64 b) { return a - b; }
U64 u64_mul(U64 a, U64 b) { return a * b; } // младшие 64 бита
U64 u64_xor(U64 a, U64 b) { return a ^ b; }
U64 u64_shr(U64 a, uint s) { return a >> s; } // s < 64
U64 u64_shl(U64 a, uint s) { return a << s; } // s < 64
bool u64_lt(U64 a, U64 b) { return a < b; }
bool u64_eq(U64 a, U64 b) { return a == b; }

#else

#define U64 uvec2
U64 u64(uint lo, uint hi) { return uvec2(lo, hi); }
uint u64_lo(U64 a) { return a.x; }
uint u64_hi(U64 a) { return a.y; }
U64 u64_add(U64 a, U64 b) {
    uint c;
    uint lo = uaddCarry(a.x, b.x, c);
    return uvec2(lo, a.y + b.y + c);
}
U64 u64_sub(U64 a, U64 b) {
    uint br;
    uint lo = usubBorrow(a.x, b.x, br);
    return uvec2(lo, a.y - b.y - br);
}
U64 u64_mul(U64 a, U64 b) {
    uint hi, lo;
    umulExtended(a.x, b.x, hi, lo);
    return uvec2(lo, hi + a.x * b.y + a.y * b.x);
}
U64 u64_xor(U64 a, U64 b) { return a ^ b; }
U64 u64_shr(U64 a, uint s) {
    if (s == 0u) return a;
    if (s >= 32u) return uvec2(a.y >> (s - 32u), 0u);
    return uvec2((a.x >> s) | (a.y << (32u - s)), a.y >> s);
}
U64 u64_shl(U64 a, uint s) {
    if (s == 0u) return a;
    if (s >= 32u) return uvec2(0u, a.x << (s - 32u));
    return uvec2(a.x << s, (a.y << s) | (a.x >> (32u - s)));
}
bool u64_lt(U64 a, U64 b) { return a.y < b.y || (a.y == b.y && a.x < b.x); }
bool u64_eq(U64 a, U64 b) { return a.x == b.x && a.y == b.y; }

#endif

U64 u64_from(uint x) { return u64(x, 0u); }
bool u64_is_zero(U64 a) { return u64_lo(a) == 0u && u64_hi(a) == 0u; }
uint u64_bit(U64 a, uint i) { return i >= 32u ? (u64_hi(a) >> (i - 32u)) & 1u : (u64_lo(a) >> i) & 1u; }

// 32×32 -> 64.
U64 u64_mul32(uint a, uint b) {
    uint hi, lo;
    umulExtended(a, b, hi, lo);
    return u64(lo, hi);
}

// Старшие 64 бита 64×64 на 32-битных лимбах — дословно mulhi64_portable (arith.h).
U64 u64_mulhi(U64 a, U64 b) {
    uint a0 = u64_lo(a), a1 = u64_hi(a), b0 = u64_lo(b), b1 = u64_hi(b);
    U64 p00 = u64_mul32(a0, b0), p01 = u64_mul32(a0, b1);
    U64 p10 = u64_mul32(a1, b0), p11 = u64_mul32(a1, b1);
    U64 mid = u64_add(u64_add(u64_from(u64_hi(p00)), u64_from(u64_lo(p01))), u64_from(u64_lo(p10)));
    return u64_add(u64_add(u64_add(p11, u64_from(u64_hi(p01))), u64_from(u64_hi(p10))),
                   u64_from(u64_hi(mid)));
}

// Число ведущих нулей, a != 0.
uint u64_clz(U64 a) {
    return u64_hi(a) != 0u ? 31u - uint(findMSB(u64_hi(a))) : 63u - uint(findMSB(u64_lo(a)));
}
uint u64_ctz(U64 a) {
    return u64_lo(a) != 0u ? uint(findLSB(u64_lo(a))) : 32u + uint(findLSB(u64_hi(a)));
}

// Деление (u1·2^32 + u0) / v при u1 < v: частное и остаток помещаются в 32 бита.
// Алгоритм divlu (Hacker's Delight, 9-3): нормализация и две 16-битные цифры
// частного, оценённые 32-битным делением с коррекцией. Только 32-битные
// операции — одинаково быстро в обоих вариантах U64.
uint divlu(uint u1, uint u0, uint v, out uint rem) {
    const uint b = 65536u;
    uint s = 31u - uint(findMSB(v));
    v <<= s;
    uint vn1 = v >> 16, vn0 = v & 0xFFFFu;
    uint un32 = (u1 << s) | (s == 0u ? 0u : (u0 >> (32u - s)));
    uint un10 = u0 << s;
    uint un1 = un10 >> 16, un0 = un10 & 0xFFFFu;
    uint q1 = un32 / vn1;
    uint rhat = un32 - q1 * vn1;
    while (q1 >= b || q1 * vn0 > b * rhat + un1) {
        q1 -= 1u;
        rhat += vn1;
        if (rhat >= b) break;
    }
    uint un21 = un32 * b + un1 - q1 * v;
    uint q0 = un21 / vn1;
    rhat = un21 - q0 * vn1;
    while (q0 >= b || q0 * vn0 > b * rhat + un0) {
        q0 -= 1u;
        rhat += vn1;
        if (rhat >= b) break;
    }
    rem = (un21 * b + un0 - q0 * v) >> s;
    return q1 * b + q0;
}

// a / d и a % d для 64-битного a и 32-битного d > 0.
U64 u64_divmod32(U64 a, uint d, out uint rem) {
    uint hi = u64_hi(a);
    uint qh = hi / d;
    uint ql = divlu(hi - qh * d, u64_lo(a), d, rem);
    return u64(ql, qh);
}

#endif // MONKEY_U64_GLSL
