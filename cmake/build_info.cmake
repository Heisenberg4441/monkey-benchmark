# Пишет gen/build_info.h: версия, git sha (с пометкой -dirty), флаги и тип
# сборки. Запускается на каждой сборке; файл переписывается только при
# изменении, чтобы не пересобирать лишнее.
# Вход: -DSRC=<корень> -DOUT=<файл> -DVERSION= -DFLAGS= -DBUILD_TYPE=
set(sha "unknown")
find_package(Git QUIET)
if(GIT_FOUND)
    execute_process(COMMAND ${GIT_EXECUTABLE} rev-parse --short=12 HEAD
                    WORKING_DIRECTORY ${SRC} OUTPUT_VARIABLE out RESULT_VARIABLE rc
                    OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
    if(rc EQUAL 0)
        set(sha "${out}")
        execute_process(COMMAND ${GIT_EXECUTABLE} diff --quiet HEAD -- src CMakeLists.txt cmake
                        WORKING_DIRECTORY ${SRC} RESULT_VARIABLE dirty ERROR_QUIET)
        if(NOT dirty EQUAL 0)
            string(APPEND sha "-dirty")
        endif()
    endif()
endif()
string(STRIP "${FLAGS}" FLAGS)
string(REGEX REPLACE "  +" " " FLAGS "${FLAGS}")
set(content "// Сгенерировано cmake/build_info.cmake — не править.
#pragma once
#define MONKEY_VERSION \"${VERSION}\"
#define MONKEY_GIT_SHA \"${sha}\"
#define MONKEY_BUILD_FLAGS \"${FLAGS}\"
#define MONKEY_BUILD_TYPE \"${BUILD_TYPE}\"
")
if(EXISTS "${OUT}")
    file(READ "${OUT}" old)
endif()
if(NOT "${old}" STREQUAL "${content}")
    file(WRITE "${OUT}" "${content}")
endif()
