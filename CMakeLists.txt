cmake_minimum_required(VERSION 3.20)

project(consequence_of_boredom_and_curiosity C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

option(COB_BUILD_TESTS "Build correctness tests" ON)
option(COB_BUILD_BENCH "Build benchmark runner" ON)
option(COB_BENCH_EXTERNAL_CBLAS "Build benchmark with an external CBLAS comparison" OFF)
option(COB_BENCH_EXTERNAL_FORTRAN_SGEMM "Build benchmark with an external Fortran SGEMM comparison" OFF)
set(COB_EXTERNAL_CBLAS_NAME "external_CBLAS" CACHE STRING "Label for the external CBLAS benchmark row")
set(COB_EXTERNAL_CBLAS_HEADER "cblas.h" CACHE STRING "Header to include for the external CBLAS benchmark")
set(COB_EXTERNAL_CBLAS_INCLUDE_DIRS "" CACHE STRING "Include directories for the external CBLAS benchmark")
set(COB_EXTERNAL_CBLAS_LIBRARIES "" CACHE STRING "Libraries for the external CBLAS benchmark")
set(COB_EXTERNAL_FORTRAN_SGEMM_NAME "external_sgemm_" CACHE STRING "Label for the external Fortran SGEMM benchmark row")
set(COB_EXTERNAL_FORTRAN_SGEMM_LIBRARIES "" CACHE STRING "Libraries for the external Fortran SGEMM benchmark")

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
endif()

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
