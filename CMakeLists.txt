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
# -march=native только если явно включён (MONKEY_NATIVE_ARCH=ON).
# По умолчанию сборка с переносимым baseline: бинарь запускается на машине,
# отличной от build-хоста, без Illegal instruction.
option(MONKEY_NATIVE_ARCH "Оптимизация под локальную микроархитектуру (-march=native)" OFF)
# Юнит-тесты (Catch2). CPU-only, без GPU SDK — собираются в любом окружении.
option(MONKEY_BUILD_TESTS "Сборка юнит-тестов (Catch2 + ctest)" ON)
# Принудительно переносимая арифметика ядер (без __int128 и интринсиков), как на
# MSVC/в GLSL. Нужна CI, чтобы проверять эти пути на clang/gcc.
option(MONKEY_PORTABLE_ARITH "Переносимые 32-битные пути арифметики ядер" OFF)

find_package(Threads REQUIRED)

# MSVC иначе перекодирует UTF-8 литералы в системную кодовую страницу (cp1251).
# Для C++ — напрямую; для CUDA — через -Xcompiler, иначе nvcc принимает /utf-8
# за свой аргумент и падает на разборе командной строки.
if(MSVC)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/utf-8> $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>)
endif()

# --- Workloads (что считает бенч) ------------------------------------------
# Статическая библиотека, общая для бинаря и тестов: описание kernel'ов и
# эталонный CPU-исполнитель (kernels.cpp) поверх header-only ядер
# (__host__ __device__) в src/workloads/.
# Эталонные цифры π встраиваются в бинарь: BBP проверяется независимо от CWD.
set(PI_HEX_HEADER ${CMAKE_BINARY_DIR}/gen/pi_hex_data.h)
add_custom_command(
    OUTPUT ${PI_HEX_HEADER}
    COMMAND ${CMAKE_COMMAND} -DIN=${CMAKE_SOURCE_DIR}/reference/pi_hex.txt
            -DOUT=${PI_HEX_HEADER} -P ${CMAKE_SOURCE_DIR}/cmake/embed_pi.cmake
    DEPENDS ${CMAKE_SOURCE_DIR}/reference/pi_hex.txt ${CMAKE_SOURCE_DIR}/cmake/embed_pi.cmake
    COMMENT "Встраивание reference/pi_hex.txt"
    VERBATIM)

add_library(monkey_workloads STATIC
    src/kernels.cpp
    src/pi_digits.cpp
    ${PI_HEX_HEADER})
target_include_directories(monkey_workloads PUBLIC src PRIVATE ${CMAKE_BINARY_DIR}/gen)
if(MONKEY_PORTABLE_ARITH)
    target_compile_definitions(monkey_workloads PUBLIC MONKEY_PORTABLE_ARITH)
endif()

# --- Vulkan-бэкенд ----------------------------------------------------------
# Библиотека monkey_vulkan: исполнитель диапазонов (vulkan_device.cpp) и SPIR-V
# src/shaders/kernels.comp в двух вариантах — uint64_t (shaderInt64) и uvec2.
# Без SDK или glslangValidator — заглушка с понятной ошибкой в рантайме.
find_package(Vulkan)
set(MONKEY_HAS_VULKAN OFF)
if(USE_VULKAN AND Vulkan_FOUND)
    if(Vulkan_GLSLANG_VALIDATOR_EXECUTABLE)
        set(GLSLANG ${Vulkan_GLSLANG_VALIDATOR_EXECUTABLE})
    else()
        find_program(GLSLANG NAMES glslangValidator HINTS ENV VULKAN_SDK PATH_SUFFIXES bin Bin)
    endif()
    if(GLSLANG)
        set(MONKEY_HAS_VULKAN ON)
    else()
        message(STATUS "Vulkan найден, но нет glslangValidator — Vulkan отключён")
    endif()
endif()

if(MONKEY_HAS_VULKAN)
    message(STATUS "Vulkan найден: GPU-бэкенд (любой GPU) включён")
    set(KERNELS_COMP ${CMAKE_SOURCE_DIR}/src/shaders/kernels.comp)
    set(KERNELS_DEPS ${KERNELS_COMP} ${CMAKE_SOURCE_DIR}/src/shaders/u64.glsl
        ${CMAKE_SOURCE_DIR}/src/shaders/arith.glsl ${CMAKE_SOURCE_DIR}/src/philox.glsl)
    set(SPV_HEADERS "")
    foreach(variant native emu)
        set(out ${CMAKE_BINARY_DIR}/gen/kernels_${variant}_spv.h)
        if(variant STREQUAL "native")
            set(defs -DU64_NATIVE)
        else()
            set(defs "")
        endif()
        add_custom_command(
            OUTPUT ${out}
            COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/gen
            COMMAND ${GLSLANG} -V --target-env vulkan1.1 ${defs} -I${CMAKE_SOURCE_DIR}/src
                    --vn kernels_${variant}_spv -o ${out} ${KERNELS_COMP}
            DEPENDS ${KERNELS_DEPS}
            COMMENT "Компиляция kernels.comp (${variant}) -> SPIR-V"
            VERBATIM)
        list(APPEND SPV_HEADERS ${out})
    endforeach()
    add_library(monkey_vulkan STATIC src/vulkan_device.cpp ${SPV_HEADERS})
    target_include_directories(monkey_vulkan PRIVATE ${CMAKE_BINARY_DIR}/gen)
    target_link_libraries(monkey_vulkan PUBLIC monkey_workloads Vulkan::Vulkan)
else()
    message(STATUS "Vulkan не найден или отключён")
    add_library(monkey_vulkan STATIC src/vulkan_stub.cpp)
    target_link_libraries(monkey_vulkan PUBLIC monkey_workloads)
endif()

# --- CUDA-бэкенд ------------------------------------------------------------
include(CheckLanguage)
check_language(CUDA)
set(MONKEY_HAS_CUDA OFF)
if(USE_CUDA AND CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(MONKEY_HAS_CUDA ON)
    message(STATUS "CUDA найден: GPU-бэкенд (NVIDIA) включён")
    add_library(monkey_cuda STATIC src/cuda_device.cu)
    target_link_libraries(monkey_cuda PUBLIC monkey_workloads)
    set_target_properties(monkey_cuda PROPERTIES
        CUDA_STANDARD 17
        # Статический CUDA-рантайм: пользователю нужен лишь драйвер NVIDIA.
        CUDA_RUNTIME_LIBRARY Static)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set_target_properties(monkey_cuda PROPERTIES CUDA_ARCHITECTURES "75-real;86-real;89-real;90-real")
    endif()
else()
    message(STATUS "CUDA не найден или отключён")
    add_library(monkey_cuda STATIC src/cuda_stub.cpp)
    target_link_libraries(monkey_cuda PUBLIC monkey_workloads)
endif()

# --- Бинарь -----------------------------------------------------------------
add_executable(monkey_bench src/main.cpp src/cpu_backend.cpp src/gpu_backend.cpp)
target_link_libraries(monkey_bench PRIVATE Threads::Threads monkey_workloads monkey_vulkan monkey_cuda)
if(MONKEY_HAS_CUDA)
    set_target_properties(monkey_bench PROPERTIES CUDA_RUNTIME_LIBRARY Static)
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

# -march=native только для нативных (не кросс, не релизных) сборок CPU. Горячие
# циклы живут в monkey_workloads (kernels.cpp), поэтому флаг — и на неё.
if(MONKEY_NATIVE_ARCH)
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("-march=native" HAS_MARCH_NATIVE)
    if(HAS_MARCH_NATIVE)
        target_compile_options(monkey_workloads PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-march=native>)
        target_compile_options(monkey_bench PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-march=native>)
    endif()
endif()
