# Тесты Catch2 (header-only, вендорится в third_party/). Цель полностью
# CPU-only — собирается без CUDA/Vulkan SDK, чтобы ctest шёл в любом окружении.

add_library(catch_main STATIC catch_main.cpp)
target_include_directories(catch_main PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/third_party)
# Сборка Catch2 один раз ради скорости компиляции тестов.
target_compile_definitions(catch_main PUBLIC CATCH_CONFIG_FAST_COMPILE)

# Хелпер для регистрации тест-бинаря: линкует Catch2, видит src/, ctest-таргет.
function(monkey_add_test name)
    add_executable(${name} ${ARGN})
    target_link_libraries(${name} PRIVATE catch_main Threads::Threads)
    target_include_directories(${name} PRIVATE ${CMAKE_SOURCE_DIR}/src)
    add_test(NAME ${name} COMMAND ${name})
endfunction()

monkey_add_test(test_sanity test_sanity.cpp)
