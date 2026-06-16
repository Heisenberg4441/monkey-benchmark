#pragma once

#include <cstdint>

#include "philox.h"

// Counter-based PRNG, общий для всех бэкендов (CPU/CUDA/Vulkan). Алгоритм —
// Philox4x32-10 (см. philox.h): stateless, тривиально параллелится и даёт
// бит-в-бит идентичные результаты на любом устройстве при одинаковых входах.
//
// Этот заголовок — тонкая обёртка, сохраняющая историческую сигнатуру
// rand_index() для бэкендов. Вся арифметика PRNG — в prng::philox.

#if defined(__CUDACC__)
#define MK_HD __host__ __device__
#else
#define MK_HD
#endif

namespace monkey {

// Индекс символа алфавита (размер n) на позиции pos для кандидата с ключом key.
// key обычно = порядковый номер попытки в потоке, seed различает потоки/запуски.
MK_HD inline int rand_index(uint32_t seed, uint32_t key, uint32_t pos, int n) {
    const uint32_t r = prng::index_stream(seed, key, pos);
    return static_cast<int>(r % static_cast<uint32_t>(n));
}

} // namespace monkey
