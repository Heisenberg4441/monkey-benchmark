#include "pi_digits.h"

#include "pi_hex_data.h" // сгенерирован CMake: kPiHexDigits, kPiHexPacked

namespace monkey {

std::size_t pi_hex_count() { return kPiHexDigits; }

int pi_hex_digit(std::size_t i) {
    const unsigned char b = kPiHexPacked[i / 2];
    return (i % 2 == 0) ? (b >> 4) : (b & 0xF);
}

uint32_t pi_hex_word(std::size_t i) {
    uint32_t w = 0;
    for (std::size_t j = 0; j < 8; ++j) w = (w << 4) | static_cast<uint32_t>(pi_hex_digit(i + j));
    return w;
}

} // namespace monkey
