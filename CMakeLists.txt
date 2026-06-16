cmake_minimum_required(VERSION 3.18)
project(monkey_benchmark LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Самодостаточные бинарники: статический MSVC-рантайм (/MT), чтобы .exe
# не требовал установленного VC++ Redistributable. CMP0091 (NEW по умолчанию
# с 3.15) включает абстракцию MSVC_RUNTIME_LIBRARY.
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
# Оптимизацию задаёт сам CMake для конфигурации Release: -O3 для GCC/Clang,
# /O2 для MSVC. Не переопределяем глобально, чтобы не затирать флаги MSVC.

# GPU-бэкенды включаются автоматически, если найден соответствующий SDK.
option(USE_CUDA "Сборка с CUDA-бэкендом (NVIDIA)" ON)
option(USE_VULKAN "Сборка с Vulkan-бэкендом (любой GPU)" ON)
# -march=native даёт лучшую производительность локально, но делает бинарник
# непортируемым. Для релизных сборок выключайте: -DNATIVE_ARCH=OFF.
option(NATIVE_ARCH "Оптимизация под локальную микроархитектуру (-march=native)" ON)
# Юнит-тесты (Catch2). CPU-only, без GPU SDK — собираются в любом окружении.
option(MONKEY_BUILD_TESTS "Сборка юнит-тестов (Catch2 + ctest)" ON)

find_package(Threads REQUIRED)

# --- Базовый таргет (CPU всегда) -------------------------------------------
add_executable(monkey_bench src/main.cpp src/cpu_backend.cpp)
target_link_libraries(monkey_bench PRIVATE Threads::Threads)

# --- CUDA-бэкенд ------------------------------------------------------------
include(CheckLanguage)
check_language(CUDA)
if(USE_CUDA AND CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    message(STATUS "CUDA найден: GPU-бэкенд (NVIDIA) включён")
    target_sources(monkey_bench PRIVATE src/cuda_backend.cu)
    target_compile_definitions(monkey_bench PRIVATE USE_CUDA)
    set_target_properties(monkey_bench PROPERTIES
        CUDA_STANDARD 17
        CUDA_SEPARABLE_COMPILATION ON
        # Статический CUDA-рантайм: пользователю нужен лишь драйвер NVIDIA.
        CUDA_RUNTIME_LIBRARY Static)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set_target_properties(monkey_bench PROPERTIES CUDA_ARCHITECTURES "75-real;86-real;89-real;90-real")
    endif()
else()
    message(STATUS "CUDA не найден или отключён")
    target_sources(monkey_bench PRIVATE src/cuda_stub.cpp)
endif()

# --- Vulkan-бэкенд ----------------------------------------------------------
find_package(Vulkan)
if(USE_VULKAN AND Vulkan_FOUND)
    # Компилятор GLSL->SPIR-V: предпочитаем glslangValidator, иначе glslc.
    if(Vulkan_GLSLANG_VALIDATOR_EXECUTABLE)
        set(GLSL_COMPILER ${Vulkan_GLSLANG_VALIDATOR_EXECUTABLE})
    else()
        find_program(GLSL_COMPILER NAMES glslangValidator glslc
                     HINTS ENV VULKAN_SDK PATH_SUFFIXES bin Bin)
    endif()
    if(GLSL_COMPILER)
        message(STATUS "Vulkan найден: GPU-бэкенд (любой GPU) включён")
        set(SPV_HEADER ${CMAKE_BINARY_DIR}/gen/monkey_comp_spv.h)
        if(GLSL_COMPILER MATCHES "glslc")
            set(GLSL_CMD ${GLSL_COMPILER} -mfmt=c -o ${SPV_HEADER}
                ${CMAKE_SOURCE_DIR}/src/monkey.comp)
        else()
            set(GLSL_CMD ${GLSL_COMPILER} -V --vn monkey_comp_spv
                -o ${SPV_HEADER} ${CMAKE_SOURCE_DIR}/src/monkey.comp)
        endif()
        add_custom_command(
            OUTPUT ${SPV_HEADER}
            COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/gen
            COMMAND ${GLSL_CMD}
            DEPENDS ${CMAKE_SOURCE_DIR}/src/monkey.comp
            COMMENT "Компиляция monkey.comp -> SPIR-V"
            VERBATIM)
        target_sources(monkey_bench PRIVATE src/vulkan_backend.cpp ${SPV_HEADER})
        target_include_directories(monkey_bench PRIVATE ${CMAKE_BINARY_DIR}/gen)
        target_compile_definitions(monkey_bench PRIVATE USE_VULKAN)
        target_link_libraries(monkey_bench PRIVATE Vulkan::Vulkan)
    else()
        message(STATUS "Vulkan найден, но нет glslangValidator/glslc — Vulkan отключён")
        target_sources(monkey_bench PRIVATE src/vulkan_stub.cpp)
    endif()
else()
    message(STATUS "Vulkan не найден или отключён")
    target_sources(monkey_bench PRIVATE src/vulkan_stub.cpp)
endif()

# --- Флаги компилятора ------------------------------------------------------
# MSVC иначе перекодирует UTF-8 литералы в системную кодовую страницу (cp1251).
# Для C++ — напрямую; для CUDA — через -Xcompiler, иначе nvcc принимает /utf-8
# за свой аргумент и падает на разборе командной строки.
if(MSVC)
    target_compile_options(monkey_bench PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/utf-8>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>)
endif()

# Linux: статически линкуем libstdc++/libgcc, чтобы бинарник не зависел от
# версии libstdc++ в дистрибутиве. Полный -static нельзя: GPU-рантаймы
# динамически грузят драйвер через dlopen.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_link_options(monkey_bench PRIVATE -static-libstdc++ -static-libgcc)
endif()

# --- Тесты ------------------------------------------------------------------
if(MONKEY_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# -march=native только для нативных (не кросс, не релизных) сборок CPU.
if(NATIVE_ARCH)
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("-march=native" HAS_MARCH_NATIVE)
    if(HAS_MARCH_NATIVE)
        target_compile_options(monkey_bench PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-march=native>)
    endif()
endif()
