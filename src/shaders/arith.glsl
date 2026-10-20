// Построчный порт src/workloads/arith.h: перемешивание для checksum и
// Montgomery-умножение (R = 2^64 и R = 2^32). Требует u64.glsl.
#ifndef MONKEY_ARITH_GLSL
#define MONKEY_ARITH_GLSL

U64 fmix64(U64 z) {
    z = u64_mul(u64_xor(z, u64_shr(z, 30u)), u64(0x1ce4e5b9u, 0xbf58476du));
    z = u64_mul(u64_xor(z, u64_shr(z, 27u)), u64(0x133111ebu, 0x94d049bbu));
    return u64_xor(z, u64_shr(z, 31u));
}

// Вклад индекса i с результатом r в checksum; r == 0 вклада не даёт.
U64 contrib(U64 i, U64 r) {
    return u64_is_zero(r) ? u64(0u, 0u) : fmix64(u64_xor(fmix64(i), r));
}

U64 inv64(U64 n) {
    U64 x = u64_xor(u64_mul(u64_from(3u), n), u64_from(2u));
    for (int i = 0; i < 4; ++i) x = u64_mul(x, u64_sub(u64_from(2u), u64_mul(n, x)));
    return x;
}

U64 mont64_mul(U64 a, U64 b, U64 n, U64 ninv) {
    U64 lo = u64_mul(a, b);
    U64 hi = u64_mulhi(a, b);
    U64 mh = u64_mulhi(u64_mul(lo, ninv), n);
    U64 r = u64_sub(hi, mh);
    return u64_lt(hi, mh) ? u64_add(r, n) : r;
}

// 2^64 mod n — двоичное деление столбиком (переносимый путь pow2_64_mod).
U64 pow2_64_mod(U64 n) {
    uint s = u64_clz(n);
    U64 x = u64_sub(u64(0u, 0u), u64_shl(n, s));
    for (int i = int(s) - 1; i >= 0; --i) {
        U64 t = u64_shl(n, uint(i));
        if (!u64_lt(x, t)) x = u64_sub(x, t);
    }
    return u64_lt(x, n) ? x : u64_sub(x, n);
}

U64 addmod64(U64 a, U64 b, U64 n) {
    U64 nb = u64_sub(n, b);
    return u64_lt(a, nb) ? u64_add(a, b) : u64_sub(a, nb);
}

uint inv32(uint m) {
    uint x = (3u * m) ^ 2u;
    for (int i = 0; i < 3; ++i) x *= 2u - m * x;
    return x;
}

uint mont32_mul(uint a, uint b, uint m, uint minv) {
    uint th, tl;
    umulExtended(a, b, th, tl);
    uint mh, ml;
    umulExtended(tl * minv, m, mh, ml);
    uint r = th - mh;
    return th < mh ? r + m : r;
}

uint addmod32(uint a, uint b, uint m) {
    return a >= m - b ? a - (m - b) : a + b;
}

#endif // MONKEY_ARITH_GLSL
