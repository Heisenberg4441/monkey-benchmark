// Дымовой тест: подтверждает, что тестовая инфраструктура (Catch2 + ctest)
// собирается и запускается. Реальные тесты PRNG появляются в test_philox.*.
#include "catch.hpp"

TEST_CASE("test harness is wired up", "[sanity]") {
    REQUIRE(1 + 1 == 2);
}
