# Встраивает reference/pi_hex.txt в бинарь: hex-цифры упаковываются по две в
# байт (0x24, 0x3f, ...). Байтовый массив, а не строковый литерал: MSVC не
# допускает литералы длиннее 64 КБ. Запуск: cmake -DIN=... -DOUT=... -P embed_pi.cmake
file(READ "${IN}" digits)
string(REGEX REPLACE "[^0-9a-fA-F]" "" digits "${digits}")
string(TOLOWER "${digits}" digits)
string(LENGTH "${digits}" count)
math(EXPR odd "${count} % 2")
if(odd)
    string(APPEND digits "0")
endif()
# 8 байт на строку (CMake regex знает только ссылки \1..\9): шаблон из
# 8 пар цифр, каждая пара -> 0xNN,
string(REPEAT "([0-9a-f][0-9a-f])" 8 row_re)
set(row_out "")
foreach(i RANGE 1 8)
    string(APPEND row_out "0x\\${i},")
endforeach()
string(REGEX REPLACE "${row_re}" "${row_out}\n" body "${digits}")
# Хвост короче 8 пар.
string(REGEX REPLACE "\n([0-9a-f][0-9a-f])" "\n0x\\1," body "${body}")
string(REGEX REPLACE ",([0-9a-f][0-9a-f])" ",0x\\1" body "${body}")
string(REGEX REPLACE "^([0-9a-f][0-9a-f])" "0x\\1," body "${body}")
file(WRITE "${OUT}" "// Сгенерировано cmake/embed_pi.cmake из reference/pi_hex.txt — не править.\n#pragma once\n#include <cstddef>\nnamespace monkey {\nconstexpr std::size_t kPiHexDigits = ${count};\nconst unsigned char kPiHexPacked[] = {\n${body}};\n} // namespace monkey\n")
