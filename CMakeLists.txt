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

# Опция CUDA. По умолчанию пытаемся включить, если Toolkit найден.
option(USE_CUDA "Сборка с CUDA-бэкендом" ON)
# -march=native даёт лучшую производительность локально, но делает бинарник
# непортируемым. Для релизных сборок выключайте: -DNATIVE_ARCH=OFF.
option(NATIVE_ARCH "Оптимизация под локальную микроархитектуру (-march=native)" ON)

include(CheckLanguage)
check_language(CUDA)

set(SOURCES
    src/main.cpp
    src/cpu_backend.cpp
)

find_package(Threads REQUIRED)

if(USE_CUDA AND CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    message(STATUS "CUDA найден: GPU-бэкенд включён")
    list(APPEND SOURCES src/cuda_backend.cu)
    add_executable(monkey_bench ${SOURCES})
    target_compile_definitions(monkey_bench PRIVATE USE_CUDA)
    set_target_properties(monkey_bench PROPERTIES
        CUDA_STANDARD 17
        CUDA_SEPARABLE_COMPILATION ON
        # Статический CUDA-рантайм: рядом с бинарником не нужны CUDA-DLL и не
        # требуется установленный Toolkit у пользователя (только драйвер NVIDIA).
        CUDA_RUNTIME_LIBRARY Static)
    # Архитектуры по умолчанию; для релизного fat-binary переопределяется
    # через -DCMAKE_CUDA_ARCHITECTURES=... (несколько -real + один -virtual/PTX).
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set_target_properties(monkey_bench PROPERTIES CUDA_ARCHITECTURES "75-real;86-real;89-real;90-real")
    endif()
    target_link_libraries(monkey_bench PRIVATE Threads::Threads)
else()
    message(STATUS "CUDA не найден или отключён: только CPU-бэкенд")
    list(APPEND SOURCES src/cuda_stub.cpp)
    add_executable(monkey_bench ${SOURCES})
    target_link_libraries(monkey_bench PRIVATE Threads::Threads)
endif()

# MSVC иначе перекодирует UTF-8 литералы в системную кодовую страницу (cp1251),
# из-за чего русский вывод превращается в кракозябры. /utf-8 фиксит источник и
# исполнение в UTF-8.
if(MSVC)
    # Для C++ — напрямую; для CUDA — через -Xcompiler, иначе nvcc принимает
    # /utf-8 за свой аргумент и падает на разборе командной строки.
    target_compile_options(monkey_bench PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/utf-8>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>)
endif()

# Linux: статически линкуем libstdc++/libgcc, чтобы бинарник не зависел от
# версии libstdc++ в дистрибутиве пользователя. Полный -static здесь нельзя:
# CUDA-рантайм динамически грузит драйвер через dlopen.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_link_options(monkey_bench PRIVATE -static-libstdc++ -static-libgcc)
endif()

# -march=native только для нативных (не кросс, не релизных) сборок CPU.
if(NATIVE_ARCH)
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("-march=native" HAS_MARCH_NATIVE)
    if(HAS_MARCH_NATIVE)
        target_compile_options(monkey_bench PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-march=native>)
    endif()
endif()
