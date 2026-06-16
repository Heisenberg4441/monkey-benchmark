// Единственная единица трансляции, которая разворачивает main() Catch2.
// Остальные тестовые файлы лишь подключают catch.hpp без этого макроса.
#define CATCH_CONFIG_MAIN
#include "catch.hpp"
