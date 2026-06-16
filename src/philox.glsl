// Philox4x32-10 для GLSL — дословный перенос src/philox.h. Подключается и
// боевым шейдером (monkey.comp), и тестовым дампом (tests/philox_dump.comp),
// поэтому обе цели гарантированно используют один и тот же код PRNG.
// Бит-в-бит совпадение с CPU проверяется тестами test_glsl_philox /
// test_vulkan_philox.
#ifndef MONKEY_PHILOX_GLSL
#define MONKEY_PHILOX_GLSL

const uint PHILOX_M0 = 0xD2511F53u;
const uint PHILOX_M1 = 0xCD9E8D57u;
const uint PHILOX_W0 = 0x9E3779B9u;
const uint PHILOX_W1 = 0xBB67AE85u;

uvec4 philox_round(uvec4 ctr, uvec2 key) {
    uint hi0, lo0, hi1, lo1;
    umulExtended(PHILOX_M0, ctr.x, hi0, lo0); // hi = старшие 32 бита
    umulExtended(PHILOX_M1, ctr.z, hi1, lo1);
    return uvec4(hi1 ^ ctr.y ^ key.x, lo1, hi0 ^ ctr.w ^ key.y, lo0);
}

uvec4 philox4x32_10(uvec4 ctr, uvec2 key) {
    for (int r = 0; r < 10; ++r) {
        if (r > 0) key += uvec2(PHILOX_W0, PHILOX_W1);
        ctr = philox_round(ctr, key);
    }
    return ctr;
}

uint index_stream(uint seed, uint key, uint pos) {
    return philox4x32_10(uvec4(key, pos, 0u, 0u), uvec2(seed, 0u)).x;
}

int rand_index(uint seed, uint key, uint pos, int n) {
    return int(index_stream(seed, key, pos) % uint(n));
}

#endif // MONKEY_PHILOX_GLSL
