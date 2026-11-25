cmake_minimum_required(VERSION 3.20)

project(consequence_of_boredom_and_curiosity C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

option(COB_BUILD_TESTS "Build correctness tests" ON)
option(COB_BUILD_BENCH "Build benchmark runner" ON)
option(COB_ENABLE_HK15 "Use one level of Hopcroft-Kerr on eligible 2-by-3-by-3 products" OFF)
option(COB_ENABLE_HK15_RAGGED "Enable HK15 with exact edge products on nondivisible shapes" OFF)
option(COB_ENABLE_APPLE_MEDIUM_WIDE "Use the SME2 medium-wide FP32 kernel on Apple arm64" OFF)
option(COB_ENABLE_N2_SMALL "Use the Neoverse N2 small-matrix kernel on Linux Arm64" OFF)
option(COB_ENABLE_X86_STREAM_B "Use runtime-dispatched packed B panels for medium-wide FP32 products on Linux x86-64" OFF)
option(COB_ENABLE_X86_MEDIUM_ROUTE "Reuse packed left panels for medium row-wide products on Linux x86-64" OFF)
option(COB_ENABLE_PORTABLE_STRASSEN "Use one-level Strassen on eligible large native x86-64 and Arm64 products" OFF)
option(COB_ENABLE_RECURSIVE_STRASSEN "Use a second Winograd level on eligible native Linux products" OFF)
option(COB_ENABLE_OPENBLAS_LEAF "Use single-thread OpenBLAS subproducts for native Linux Strassen builds" OFF)
option(COB_ENABLE_ALIGNED_WINOGRAD "Align two-level rectangular Winograd subproducts to the native leaf tiles" OFF)
set(COB_BLAS_LEAF "NONE" CACHE STRING "Strassen leaf provider (NONE, OPENBLAS, MKL or ARMPL)")
set_property(CACHE COB_BLAS_LEAF PROPERTY STRINGS NONE OPENBLAS MKL ARMPL)
set(COB_BLAS_ROOT "" CACHE PATH "Installation prefix of the sequential oneMKL or ArmPL library")
if(NOT COB_BLAS_LEAF MATCHES "^(NONE|OPENBLAS|MKL|ARMPL)$")
    message(FATAL_ERROR "COB_BLAS_LEAF must be NONE, OPENBLAS, MKL or ARMPL")
endif()
if(COB_ENABLE_OPENBLAS_LEAF)
    if(NOT COB_BLAS_LEAF MATCHES "^(NONE|OPENBLAS)$")
        message(FATAL_ERROR "COB_ENABLE_OPENBLAS_LEAF=ON conflicts with COB_BLAS_LEAF=${COB_BLAS_LEAF}")
    endif()
    set(COB_BLAS_LEAF OPENBLAS)
endif()

function(cob_configure_blas_leaf)
    if(COB_ENABLE_ALIGNED_WINOGRAD AND (NOT COB_ENABLE_PORTABLE_STRASSEN OR
       NOT COB_ENABLE_RECURSIVE_STRASSEN OR COB_BLAS_LEAF STREQUAL "NONE"))
        message(FATAL_ERROR "COB_ENABLE_ALIGNED_WINOGRAD requires portable Strassen, recursive Strassen and an external BLAS leaf")
    endif()
    if(COB_BLAS_LEAF STREQUAL "NONE")
        return()
    endif()
    if(NOT COB_ENABLE_PORTABLE_STRASSEN OR NOT CMAKE_SYSTEM_NAME STREQUAL "Linux" OR
       NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64|aarch64|arm64|ARM64|AARCH64)$")
        message(FATAL_ERROR "An external leaf requires native Linux x86-64 or Arm64 and COB_ENABLE_PORTABLE_STRASSEN=ON")
    endif()
    add_library(cob_blas_leaf INTERFACE)
    if(COB_ENABLE_ALIGNED_WINOGRAD)
        target_compile_definitions(cob_blas_leaf INTERFACE COB_ENABLE_ALIGNED_WINOGRAD=1)
    endif()
    if(COB_BLAS_LEAF STREQUAL "OPENBLAS")
        find_package(PkgConfig REQUIRED)
        pkg_check_modules(COB_OPENBLAS REQUIRED IMPORTED_TARGET openblas)
        target_compile_definitions(cob_blas_leaf INTERFACE COB_ENABLE_OPENBLAS_LEAF=1)
        target_link_libraries(cob_blas_leaf INTERFACE PkgConfig::COB_OPENBLAS)
        return()
    endif()

    # These providers expose distinct LP64 sequential components; only their
    # shared libraries support the runtime identity check.
    if(COB_BLAS_LEAF STREQUAL "MKL")
        if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64)$")
            message(FATAL_ERROR "The oneMKL leaf requires Linux x86-64")
        endif()
        set(prefix "$ENV{MKLROOT}")
        set(header mkl.h)
        set(components mkl_intel_lp64 mkl_sequential mkl_core)
        target_compile_definitions(cob_blas_leaf INTERFACE COB_ENABLE_MKL_SEQUENTIAL_LEAF=1)
    else()
        if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64|ARM64|AARCH64)$")
            message(FATAL_ERROR "The ArmPL leaf requires Linux Arm64")
        endif()
        set(prefix "$ENV{ARMPL_DIR}")
        set(header armpl.h)
        set(components armpl_lp64)
        target_compile_definitions(cob_blas_leaf INTERFACE COB_ENABLE_ARMPL_SEQUENTIAL_LEAF=1)
    endif()
    if(COB_BLAS_ROOT)
        set(prefix "${COB_BLAS_ROOT}")
    endif()
    if(prefix)
        set(search_paths PATHS "${prefix}" NO_DEFAULT_PATH)
    endif()
    # An explicit prefix selects every component from the same installation.
    # Refresh cached discovery when a build directory is reconfigured.
    unset(cob_blas_include CACHE)
    find_path(cob_blas_include NAMES "${header}" ${search_paths} PATH_SUFFIXES include REQUIRED)
    target_include_directories(cob_blas_leaf SYSTEM INTERFACE "${cob_blas_include}")
    set(CMAKE_FIND_LIBRARY_SUFFIXES .so .so.3)
    foreach(component IN LISTS components)
        unset(cob_blas_component CACHE)
        find_library(cob_blas_component NAMES "${component}" ${search_paths}
                     PATH_SUFFIXES lib lib/intel64 lib64 REQUIRED)
        list(APPEND libraries "${cob_blas_component}")
    endforeach()
    if(COB_BLAS_LEAF STREQUAL "MKL")
        # The LP64 dispatcher needs both auxiliary DSOs even with --as-needed.
        find_package(Threads REQUIRED)
        target_link_libraries(cob_blas_leaf INTERFACE -Wl,--push-state,--no-as-needed
                              ${libraries} -Wl,--pop-state Threads::Threads m ${CMAKE_DL_LIBS})
    else()
        target_link_libraries(cob_blas_leaf INTERFACE ${libraries} m ${CMAKE_DL_LIBS})
    endif()
endfunction()
cob_configure_blas_leaf()
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
            if(COB_ENABLE_APPLE_MEDIUM_WIDE)
                target_compile_definitions(cob_strassen_sme_leaf PRIVATE COB_ENABLE_APPLE_MEDIUM_WIDE=1)
            endif()
            return()
        endif()
    endif()
    if(COB_STRASSEN_SME STREQUAL "ON")
        message(FATAL_ERROR "COB_STRASSEN_SME=ON needs Apple arm64 and a compiler supporting the SME2 leaf")
    endif()
endfunction()
cob_configure_strassen_leaf()

if(COB_ENABLE_X86_STREAM_B AND CMAKE_SYSTEM_NAME STREQUAL "Linux" AND
   CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64)$" AND CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    add_library(cob_stream_b_avx2 OBJECT src/cob_stream_b_kernel.c)
    add_library(cob_stream_b_avx512 OBJECT src/cob_stream_b_kernel.c)
    # A conservative SGEMM translation unit and runtime feature checks keep the
    # archive usable on x86-64 CPUs without either instruction set.
    target_compile_options(cob_stream_b_avx2 PRIVATE -march=x86-64 -mavx2 -mfma)
    target_compile_options(cob_stream_b_avx512 PRIVATE -march=x86-64 -mavx512f -mfma)
    target_compile_definitions(cob_stream_b_avx2 PRIVATE COB_STREAM_B_AVX512=0)
    target_compile_definitions(cob_stream_b_avx512 PRIVATE COB_STREAM_B_AVX512=1)
    set(cob_stream_b_objects $<TARGET_OBJECTS:cob_stream_b_avx2> $<TARGET_OBJECTS:cob_stream_b_avx512>)
endif()

add_library(cob_matmul STATIC
    src/gemm.c
)

include(GNUInstallDirs)
install(TARGETS cob_matmul ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(FILES include/cob_gemm.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

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
    if(COB_ENABLE_APPLE_MEDIUM_WIDE AND TARGET cob_strassen_sme_leaf)
        add_executable(cob_apple_medium_wide_test tests/test_apple_medium_wide.c)
        target_link_libraries(cob_apple_medium_wide_test PRIVATE cob_matmul)
        add_test(NAME cob_apple_medium_wide_test COMMAND cob_apple_medium_wide_test)
    endif()
    if(TARGET cob_stream_b_avx2)
        add_executable(cob_stream_b_test tests/test_stream_b.c ${cob_stream_b_objects})
        target_include_directories(cob_stream_b_test PRIVATE include)
        target_compile_definitions(cob_stream_b_test PRIVATE COB_ENABLE_X86_STREAM_B=1)
        target_link_libraries(cob_stream_b_test PRIVATE m)
        add_test(NAME cob_stream_b_test COMMAND cob_stream_b_test)
    endif()

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
        if(TARGET cob_blas_leaf AND COB_ENABLE_RECURSIVE_STRASSEN)
            add_executable(cob_openblas_winograd_test tests/test_openblas_winograd.c)
            target_include_directories(cob_openblas_winograd_test PRIVATE include)
            target_compile_definitions(cob_openblas_winograd_test PRIVATE
                COB_ENABLE_PORTABLE_STRASSEN=1 COB_ENABLE_RECURSIVE_STRASSEN=1)
            if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm64|aarch64|ARM64|AARCH64)$")
                target_compile_definitions(cob_openblas_winograd_test PRIVATE COB_USE_NEON=1)
            endif()
            target_link_libraries(cob_openblas_winograd_test PRIVATE cob_blas_leaf m)
            add_test(NAME cob_openblas_winograd_test COMMAND cob_openblas_winograd_test)
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
        if(COB_ENABLE_HK15 OR COB_ENABLE_HK15_RAGGED)
            target_compile_definitions(${target} PRIVATE COB_ENABLE_HK15=1)
        endif()
        if(COB_ENABLE_HK15_RAGGED)
            target_compile_definitions(${target} PRIVATE COB_ENABLE_HK15_RAGGED=1)
            target_sources(${target} PRIVATE src/cob_hk15_ragged.c)
        endif()
        if(COB_ENABLE_APPLE_MEDIUM_WIDE AND TARGET cob_strassen_sme_leaf)
            target_compile_definitions(${target} PRIVATE COB_ENABLE_APPLE_MEDIUM_WIDE=1)
        endif()
        if(COB_ENABLE_N2_SMALL)
            target_compile_definitions(${target} PRIVATE COB_ENABLE_N2_SMALL=1)
        endif()
        if(TARGET cob_stream_b_avx2)
            target_compile_definitions(${target} PRIVATE COB_ENABLE_X86_STREAM_B=1)
            target_sources(${target} PRIVATE ${cob_stream_b_objects})
        endif()
        if(COB_ENABLE_X86_MEDIUM_ROUTE)
            target_compile_definitions(${target} PRIVATE COB_ENABLE_X86_MEDIUM_ROUTE=1)
        endif()
        if(TARGET cob_strassen_sme_leaf)
            target_sources(${target} PRIVATE $<TARGET_OBJECTS:cob_strassen_sme_leaf>)
            target_compile_definitions(${target} PRIVATE COB_WITH_STRASSEN_SME_LEAF=1)
        endif()
        if(COB_ENABLE_PORTABLE_STRASSEN)
            target_compile_definitions(${target} PRIVATE COB_ENABLE_PORTABLE_STRASSEN=1)
            if(TARGET cob_blas_leaf)
                target_link_libraries(${target} PRIVATE cob_blas_leaf)
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
