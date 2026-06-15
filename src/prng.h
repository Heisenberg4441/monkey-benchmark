#pragma once

#include <cstdint>

// Счётчик-based PRNG, общий для всех бэкендов (CPU/CUDA/Vulkan). Полностью
// stateless: значение определяется входом, без хранимого состояния — поэтому
// тривиально параллелится и даёт идентичные результаты на любом устройстве.
//
// Намеренно 32-битный (без int64): те же строки дословно переносятся в
// GLSL-шейдер, где 64-битная арифметика — необязательное расширение.
// Качество — обычный PCG-hash: для throughput-бенчмарка (Монте-Карло с
// повторами) более чем достаточно.

#if defined(__CUDACC__)
#define MK_HD __host__ __device__
#else
#define MK_HD
#endif

namespace monkey {

// PCG hash (Hash без состояния, O'Neill). Хорошая лавинность на 32 битах.
MK_HD inline uint32_t pcg_hash(uint32_t x) {
    uint32_t state = x * 747796405u + 2891336453u;
    uint32_t word = ((state >> ((state >> 28) + 4u)) ^ state) * 277803737u;
    return (word >> 22) ^ word;
}

// Индекс символа алфавита (размер n) на позиции pos для кандидата с ключом key.
// key обычно = порядковый номер попытки в потоке, seed различает потоки/запуски.
MK_HD inline int rand_index(uint32_t seed, uint32_t key, uint32_t pos, int n) {
    uint32_t r = pcg_hash(seed ^ pcg_hash(key + pos * 2654435761u));
    return static_cast<int>(r % static_cast<uint32_t>(n));
}

} // namespace monkey
