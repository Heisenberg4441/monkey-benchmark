#pragma once

#include <cstddef>
#include <cstdint>

// Эталонные hex-цифры π, встроенные в бинарь при сборке из reference/pi_hex.txt
// (cmake/embed_pi.cmake). Не зависят от текущей директории запуска.

namespace monkey {

// Число встроенных цифр (позиции 0..count-1 после «3.»).
std::size_t pi_hex_count();

// Цифра на позиции i (0..15). Требует i < pi_hex_count().
int pi_hex_digit(std::size_t i);

// 8 цифр начиная с позиции i, старшая — цифра i. Требует i + 8 <= pi_hex_count().
uint32_t pi_hex_word(std::size_t i);

} // namespace monkey
