cmake_minimum_required(VERSION 3.20)

project(consequence_of_boredom_and_curiosity C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

option(COB_BUILD_TESTS "Build correctness tests" ON)
option(COB_BUILD_BENCH "Build benchmark runner" ON)
option(COB_ENABLE_PORTABLE_STRASSEN "Use one-level Strassen on eligible large native x86-64 and Arm64 products" OFF)
option(COB_ENABLE_RECURSIVE_STRASSEN "Use a second Winograd level on eligible exact squares in native Linux Strassen builds" OFF)
option(COB_ENABLE_OPENBLAS_LEAF "Use single-thread OpenBLAS subproducts for native Linux Strassen builds" OFF)
if(COB_ENABLE_OPENBLAS_LEAF)
    if(NOT COB_ENABLE_PORTABLE_STRASSEN OR NOT CMAKE_SYSTEM_NAME STREQUAL "Linux" OR
       NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64|aarch64|arm64|ARM64|AARCH64)$")
        message(FATAL_ERROR "COB_ENABLE_OPENBLAS_LEAF=ON requires a native Linux x86-64 or Arm64 target and COB_ENABLE_PORTABLE_STRASSEN=ON")
    endif()
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(COB_OPENBLAS REQUIRED IMPORTED_TARGET openblas)
endif()
option(COB_BENCH_EXTERNAL_CBLAS "Build benchmark with an external CBLAS comparison" OFF)
option(COB_BENCH_EXTERNAL_FORTRAN_SGEMM "Build benchmark with an external Fortran SGEMM comparison" OFF)
set(COB_EXTERNAL_CBLAS_NAME "external_CBLAS" CACHE STRING "Label for the external CBLAS benchmark row")
set(COB_EXTERNAL_CBLAS_HEADER "cblas.h" CACHE STRING "Header to include for the external CBLAS benchmark")
set(COB_EXTERNAL_CBLAS_INCLUDE_DIRS "" CACHE STRING "Include directories for the external CBLAS benchmark")
set(COB_EXTERNAL_CBLAS_LIBRARIES "" CACHE STRING "Libraries for the external CBLAS benchmark")
set(COB_EXTERNAL_FORTRAN_SGEMM_NAME "external_sgemm_" CACHE STRING "Label for the external Fortran SGEMM benchmark row")
set(COB_EXTERNAL_FORTRAN_SGEMM_LIBRARIES "" CACHE STRING "Libraries for the external Fortran SGEMM benchmark")

set(COB_STRASSEN_SME "AUTO" CACHE STRING "Apple SME2 Strassen leaf (AUTO, ON or OFF)")
set_property(CACHE COB_STRASSEN_SME PROPERTY STRINGS AUTO ON OFF)
if(NOT COB_STRASSEN_SME MATCHES "^(AUTO|ON|OFF)$")
    message(FATAL_ERROR "COB_STRASSEN_SME must be AUTO, ON or OFF")
endif()

function(cob_configure_strassen_leaf)
    if(COB_STRASSEN_SME STREQUAL "OFF")
        return()
    endif()
    set(sme_source "${PROJECT_SOURCE_DIR}/third_party/maxmulsk/strassen_leaf.c")
    set(sme_flags -mcpu=apple-m4 -march=armv8.7-a+sme+sme2)
    if(CMAKE_SYSTEM_NAME STREQUAL "Darwin" AND
       (CMAKE_OSX_ARCHITECTURES STREQUAL "arm64" OR
        (NOT CMAKE_OSX_ARCHITECTURES AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm64|aarch64|ARM64|AARCH64)$")))
        set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
        try_compile(sme_compiles "${CMAKE_CURRENT_BINARY_DIR}/check_strassen_sme"
            "${sme_source}" COMPILE_DEFINITIONS ${sme_flags})
        if(sme_compiles)
            add_library(cob_strassen_sme_leaf OBJECT "${sme_source}")
            target_compile_options(cob_strassen_sme_leaf PRIVATE ${sme_flags})
            return()
        endif()
    endif()
    if(COB_STRASSEN_SME STREQUAL "ON")
        message(FATAL_ERROR "COB_STRASSEN_SME=ON needs Apple arm64 and a compiler supporting the SME2 leaf")
    endif()
endfunction()
cob_configure_strassen_leaf()

add_library(cob_matmul STATIC
    src/gemm.c
)

target_include_directories(cob_matmul
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

target_compile_options(cob_matmul
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

if(APPLE)
    target_compile_definitions(cob_matmul PRIVATE _DARWIN_C_SOURCE)
endif()

if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm64|aarch64|ARM64|AARCH64)$")
    target_compile_definitions(cob_matmul PRIVATE COB_USE_NEON=1)
endif()

if(COB_BUILD_TESTS)
    enable_testing()

    add_executable(cob_gemm_test
        tests/test_gemm.c
    )
    target_link_libraries(cob_gemm_test PRIVATE cob_matmul)

    add_test(NAME cob_gemm_test COMMAND cob_gemm_test)

    add_library(cob_matmul_strassen_test STATIC
        src/gemm.c
    )
    target_include_directories(cob_matmul_strassen_test
        PUBLIC
            ${PROJECT_SOURCE_DIR}/include
    )
    target_compile_options(cob_matmul_strassen_test
        PRIVATE
            -Wall
            -Wextra
            -Wpedantic
    )
    target_compile_definitions(cob_matmul_strassen_test PRIVATE
        COB_SGEMM_STRASSEN1_MIN_DIM=1024
    )
    if(APPLE)
        target_compile_definitions(cob_matmul_strassen_test PRIVATE _DARWIN_C_SOURCE)
    endif()
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm64|aarch64|ARM64|AARCH64)$")
        target_compile_definitions(cob_matmul_strassen_test PRIVATE COB_USE_NEON=1)
    endif()

    add_executable(cob_gemm_strassen_test
        tests/test_gemm.c
    )
    target_link_libraries(cob_gemm_strassen_test PRIVATE cob_matmul_strassen_test)

    add_test(NAME cob_gemm_strassen_test COMMAND cob_gemm_strassen_test)

    add_library(cob_matmul_strassen_square_test STATIC
        src/gemm.c
    )
    target_include_directories(cob_matmul_strassen_square_test PUBLIC ${PROJECT_SOURCE_DIR}/include)
    target_compile_options(cob_matmul_strassen_square_test PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_definitions(cob_matmul_strassen_square_test PRIVATE
        COB_SGEMM_STRASSEN1_MIN_DIM=8192
        COB_SGEMM_STRASSEN1_SQUARE_MIN_DIM=128
    )
    if(APPLE)
        target_compile_definitions(cob_matmul_strassen_square_test PRIVATE _DARWIN_C_SOURCE)
    endif()
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm64|aarch64|ARM64|AARCH64)$")
        target_compile_definitions(cob_matmul_strassen_square_test PRIVATE COB_USE_NEON=1)
    endif()

    add_executable(cob_gemm_strassen_square_test tests/test_gemm.c)
    target_link_libraries(cob_gemm_strassen_square_test PRIVATE cob_matmul_strassen_square_test)
    add_test(NAME cob_gemm_strassen_square_test COMMAND cob_gemm_strassen_square_test)
    if(COB_ENABLE_PORTABLE_STRASSEN AND UNIX AND CMAKE_C_COMPILER_ID MATCHES "Clang|GNU" AND
       CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64|arm64|aarch64|ARM64|AARCH64)$")
        add_executable(cob_portable_strassen_test tests/test_portable_strassen.c)
        target_include_directories(cob_portable_strassen_test PRIVATE include)
        target_compile_definitions(cob_portable_strassen_test PRIVATE
            COB_ENABLE_PORTABLE_STRASSEN=1 COB_PORTABLE_STRASSEN_MIN_DIM=128 COB_RECTANGULAR_MIN_HARMONIC_DIM=192
            COB_DISABLE_APPLE_AMX=1 COB_DISABLE_APPLE_SME=1)
        if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm64|aarch64|ARM64|AARCH64)$")
            target_compile_definitions(cob_portable_strassen_test PRIVATE COB_USE_NEON=1)
        endif()
        if(COB_ENABLE_RECURSIVE_STRASSEN AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
            target_compile_definitions(cob_portable_strassen_test PRIVATE
                COB_ENABLE_RECURSIVE_STRASSEN=1 COB_RECURSIVE_MIN_DIM=128 COB_RECURSIVE_MAX_DIM=256)
        endif()
        target_link_libraries(cob_portable_strassen_test PRIVATE m)
        add_test(NAME cob_portable_strassen_test COMMAND cob_portable_strassen_test)
        if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
            add_executable(cob_fused_strassen_test tests/test_fused_strassen.c)
            target_include_directories(cob_fused_strassen_test PRIVATE include)
            target_link_libraries(cob_fused_strassen_test PRIVATE m)
            add_test(NAME cob_fused_strassen_test COMMAND cob_fused_strassen_test)
            set_tests_properties(cob_fused_strassen_test PROPERTIES SKIP_RETURN_CODE 77)
        endif()
    endif()
endif()

foreach(target cob_matmul cob_matmul_strassen_test cob_matmul_strassen_square_test)
    if(TARGET ${target})
        if(TARGET cob_strassen_sme_leaf)
            target_sources(${target} PRIVATE $<TARGET_OBJECTS:cob_strassen_sme_leaf>)
            target_compile_definitions(${target} PRIVATE COB_WITH_STRASSEN_SME_LEAF=1)
        endif()
        if(COB_ENABLE_PORTABLE_STRASSEN)
            target_compile_definitions(${target} PRIVATE COB_ENABLE_PORTABLE_STRASSEN=1)
            if(COB_ENABLE_OPENBLAS_LEAF)
                target_compile_definitions(${target} PRIVATE COB_ENABLE_OPENBLAS_LEAF=1)
                target_link_libraries(${target} PRIVATE PkgConfig::COB_OPENBLAS)
            endif()
            if(COB_ENABLE_RECURSIVE_STRASSEN AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
                target_compile_definitions(${target} PRIVATE COB_ENABLE_RECURSIVE_STRASSEN=1)
            endif()
        endif()
    endif()
endforeach()

if(COB_BUILD_BENCH)
    add_executable(cob_gemm_bench
        bench/bench_gemm.c
    )
    target_link_libraries(cob_gemm_bench PRIVATE cob_matmul)

    if(APPLE)
        target_compile_definitions(cob_gemm_bench PRIVATE _DARWIN_C_SOURCE)
    endif()

    if(COB_BENCH_EXTERNAL_CBLAS)
        target_compile_definitions(cob_gemm_bench PRIVATE
            COB_HAVE_EXTERNAL_CBLAS=1
            COB_EXTERNAL_CBLAS_NAME="${COB_EXTERNAL_CBLAS_NAME}"
            COB_EXTERNAL_CBLAS_HEADER="${COB_EXTERNAL_CBLAS_HEADER}"
        )
        if(COB_EXTERNAL_CBLAS_INCLUDE_DIRS)
            target_include_directories(cob_gemm_bench SYSTEM PRIVATE ${COB_EXTERNAL_CBLAS_INCLUDE_DIRS})
        endif()
        if(COB_EXTERNAL_CBLAS_LIBRARIES)
            target_link_libraries(cob_gemm_bench PRIVATE ${COB_EXTERNAL_CBLAS_LIBRARIES})
        endif()
    elseif(COB_BENCH_EXTERNAL_FORTRAN_SGEMM)
        target_compile_definitions(cob_gemm_bench PRIVATE
            COB_HAVE_EXTERNAL_FORTRAN_SGEMM=1
            COB_EXTERNAL_FORTRAN_SGEMM_NAME="${COB_EXTERNAL_FORTRAN_SGEMM_NAME}"
        )
        if(COB_EXTERNAL_FORTRAN_SGEMM_LIBRARIES)
            target_link_libraries(cob_gemm_bench PRIVATE ${COB_EXTERNAL_FORTRAN_SGEMM_LIBRARIES})
        endif()
    elseif(APPLE)
        find_library(ACCELERATE_FRAMEWORK Accelerate)
        if(ACCELERATE_FRAMEWORK)
            target_link_libraries(cob_gemm_bench PRIVATE ${ACCELERATE_FRAMEWORK})
            target_compile_definitions(cob_gemm_bench PRIVATE
                COB_HAVE_ACCELERATE=1
                ACCELERATE_NEW_LAPACK=1
            )
        endif()
    endif()
endif()
