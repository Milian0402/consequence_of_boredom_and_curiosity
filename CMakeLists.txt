cmake_minimum_required(VERSION 3.20)

project(consequence_of_boredom_and_curiosity C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

option(COB_BUILD_TESTS "Build correctness tests" ON)
option(COB_BUILD_BENCH "Build benchmark runner" ON)

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
endif()

if(COB_BUILD_BENCH)
    add_executable(cob_gemm_bench
        bench/bench_gemm.c
    )
    target_link_libraries(cob_gemm_bench PRIVATE cob_matmul)

    if(APPLE)
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
