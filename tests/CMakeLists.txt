# Тесты Catch2 (header-only, вендорится в third_party/). Цель полностью
# CPU-only — собирается без CUDA/Vulkan SDK, чтобы ctest шёл в любом окружении.

add_library(catch_main STATIC catch_main.cpp)
target_include_directories(catch_main PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/third_party)
# Сборка Catch2 один раз ради скорости компиляции тестов.
target_compile_definitions(catch_main PUBLIC CATCH_CONFIG_FAST_COMPILE)

# Хелпер для регистрации тест-бинаря: линкует Catch2, видит src/, ctest-таргет.
function(monkey_add_test name)
    add_executable(${name} ${ARGN})
    target_link_libraries(${name} PRIVATE catch_main Threads::Threads monkey_workloads)
    target_include_directories(${name} PRIVATE ${CMAKE_SOURCE_DIR}/src)
    add_test(NAME ${name} COMMAND ${name})
endfunction()

monkey_add_test(test_sanity test_sanity.cpp)
monkey_add_test(test_philox test_philox.cpp)
monkey_add_test(test_glsl_philox test_glsl_philox.cpp)
monkey_add_test(test_counters test_counters.cpp)
monkey_add_test(test_workload_monkey test_workload_monkey.cpp)
monkey_add_test(test_uint128 test_uint128.cpp)

# --- Слой 2: реальный прогон SPIR-V на Vulkan-устройстве --------------------
# Собирается только если есть Vulkan SDK и компилятор GLSL. В рантайме тест сам
# делает WARN+pass, если устройства нет (см. test_vulkan_philox.cpp). В CI
# устройство даёт Mesa lavapipe.
find_package(Vulkan QUIET)
if(Vulkan_FOUND)
    if(Vulkan_GLSLANG_VALIDATOR_EXECUTABLE)
        set(TEST_GLSL_COMPILER ${Vulkan_GLSLANG_VALIDATOR_EXECUTABLE})
    else()
        find_program(TEST_GLSL_COMPILER NAMES glslangValidator glslc
                     HINTS ENV VULKAN_SDK PATH_SUFFIXES bin Bin)
    endif()
endif()

if(Vulkan_FOUND AND TEST_GLSL_COMPILER)
    set(DUMP_SPV ${CMAKE_CURRENT_BINARY_DIR}/gen/philox_dump_spv.h)
    if(TEST_GLSL_COMPILER MATCHES "glslc")
        set(DUMP_CMD ${TEST_GLSL_COMPILER} -mfmt=c -I ${CMAKE_SOURCE_DIR}/src
            -o ${DUMP_SPV} ${CMAKE_CURRENT_SOURCE_DIR}/philox_dump.comp)
    else()
        set(DUMP_CMD ${TEST_GLSL_COMPILER} -V -I${CMAKE_SOURCE_DIR}/src
            --vn philox_dump_spv -o ${DUMP_SPV} ${CMAKE_CURRENT_SOURCE_DIR}/philox_dump.comp)
    endif()
    add_custom_command(
        OUTPUT ${DUMP_SPV}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/gen
        COMMAND ${DUMP_CMD}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/philox_dump.comp ${CMAKE_SOURCE_DIR}/src/philox.glsl
        COMMENT "Компиляция philox_dump.comp -> SPIR-V"
        VERBATIM)
    monkey_add_test(test_vulkan_philox test_vulkan_philox.cpp ${DUMP_SPV})
    target_include_directories(test_vulkan_philox PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/gen)
    target_link_libraries(test_vulkan_philox PRIVATE Vulkan::Vulkan)
else()
    message(STATUS "Vulkan/glslang не найдены — real-shader determinism test пропущен")
endif()

# --- Cross-backend determinism на CUDA -------------------------------------
# CMAKE_CUDA_COMPILER выставляется, если родительский CMake включил язык CUDA
# (USE_CUDA + найден nvcc). Тест собирается только тогда; в рантайме сам
# делает WARN+pass без устройства.
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    monkey_add_test(test_cuda_philox test_cuda_philox.cu)
    set_target_properties(test_cuda_philox PROPERTIES CUDA_STANDARD 17)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set_target_properties(test_cuda_philox PROPERTIES
            CUDA_ARCHITECTURES "75-real;86-real;89-real;90-real")
    endif()
else()
    message(STATUS "CUDA не включён — cross-backend determinism test (CUDA) пропущен")
endif()
