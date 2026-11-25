CC ?= cc
BUILD_DIR ?= build

UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)

CFLAGS ?= -O3
CFLAGS += -std=c11 -Wall -Wextra -Wpedantic -Iinclude

ifeq ($(UNAME_S),Darwin)
CFLAGS += -D_DARWIN_C_SOURCE
BENCH_LDFLAGS += -framework Accelerate
BENCH_CFLAGS += -DCOB_HAVE_ACCELERATE=1 -DACCELERATE_NEW_LAPACK=1
endif

ifeq ($(UNAME_M),arm64)
CFLAGS += -DCOB_USE_NEON=1
endif
ifeq ($(UNAME_M),aarch64)
CFLAGS += -DCOB_USE_NEON=1
endif

# Strassen half-products can use a separately compiled SME2 leaf on Apple arm64.
# AUTO retains the native C path when the target compiler lacks SME intrinsics.
COB_STRASSEN_SME ?= AUTO
COB_ENABLE_PORTABLE_STRASSEN ?= OFF
COB_ENABLE_RECURSIVE_STRASSEN ?= OFF
COB_ENABLE_OPENBLAS_LEAF ?= OFF
COB_BLAS_LEAF ?= NONE
COB_ENABLE_ALIGNED_WINOGRAD ?= OFF
COB_ENABLE_HK15 ?= OFF
COB_ENABLE_APPLE_MEDIUM_WIDE ?= OFF
COB_ENABLE_N2_SMALL ?= OFF
COB_ENABLE_X86_MEDIUM_ROUTE ?= OFF
COB_ENABLE_X86_STREAM_B ?= OFF
COB_SME_CC ?= $(CC)
SME_SRC := third_party/maxmulsk/strassen_leaf.c
SME_FLAGS := -mcpu=apple-m4 -march=armv8.7-a+sme+sme2
ifneq ($(filter $(COB_STRASSEN_SME),AUTO ON OFF),$(COB_STRASSEN_SME))
$(error COB_STRASSEN_SME must be AUTO, ON or OFF)
endif

ifneq ($(COB_STRASSEN_SME),OFF)
ifeq ($(UNAME_S)-$(UNAME_M),Darwin-arm64)
SME_COMPILER_OK := $(shell $(COB_SME_CC) $(CFLAGS) $(SME_FLAGS) -fsyntax-only $(SME_SRC) >/dev/null 2>&1 && echo yes)
endif
ifeq ($(SME_COMPILER_OK),yes)
SME_OBJ := $(BUILD_DIR)/strassen_sme_leaf.o
SME_DEFINE := -DCOB_WITH_STRASSEN_SME_LEAF=1
else ifeq ($(COB_STRASSEN_SME),ON)
$(error COB_STRASSEN_SME=ON needs Apple arm64 and a compiler supporting the SME2 leaf)
endif
endif

ifeq ($(COB_ENABLE_PORTABLE_STRASSEN),ON)
SME_DEFINE += -DCOB_ENABLE_PORTABLE_STRASSEN=1
else ifneq ($(COB_ENABLE_PORTABLE_STRASSEN),OFF)
$(error COB_ENABLE_PORTABLE_STRASSEN must be ON or OFF)
endif

ifeq ($(COB_ENABLE_RECURSIVE_STRASSEN),ON)
SME_DEFINE += -DCOB_ENABLE_RECURSIVE_STRASSEN=1
else ifneq ($(COB_ENABLE_RECURSIVE_STRASSEN),OFF)
$(error COB_ENABLE_RECURSIVE_STRASSEN must be ON or OFF)
endif

ifneq ($(words $(COB_BLAS_LEAF)),1)
$(error COB_BLAS_LEAF must select exactly one provider)
endif
ifeq ($(filter NONE OPENBLAS MKL ARMPL,$(COB_BLAS_LEAF)),)
$(error COB_BLAS_LEAF must be NONE, OPENBLAS, MKL or ARMPL)
endif
leaf_provider := $(COB_BLAS_LEAF)
ifeq ($(COB_ENABLE_OPENBLAS_LEAF),ON)
ifeq ($(filter $(leaf_provider),NONE OPENBLAS),)
$(error COB_ENABLE_OPENBLAS_LEAF=ON conflicts with COB_BLAS_LEAF=$(leaf_provider))
endif
leaf_provider := OPENBLAS
else ifneq ($(COB_ENABLE_OPENBLAS_LEAF),OFF)
$(error COB_ENABLE_OPENBLAS_LEAF must be ON or OFF)
endif

ifeq ($(COB_ENABLE_ALIGNED_WINOGRAD),ON)
ifneq ($(COB_ENABLE_PORTABLE_STRASSEN):$(COB_ENABLE_RECURSIVE_STRASSEN),ON:ON)
$(error COB_ENABLE_ALIGNED_WINOGRAD requires portable Strassen and recursive Strassen)
endif
ifeq ($(leaf_provider),NONE)
$(error COB_ENABLE_ALIGNED_WINOGRAD requires an external BLAS leaf)
endif
SME_DEFINE += -DCOB_ENABLE_ALIGNED_WINOGRAD=1
else ifneq ($(COB_ENABLE_ALIGNED_WINOGRAD),OFF)
$(error COB_ENABLE_ALIGNED_WINOGRAD must be ON or OFF)
endif

ifneq ($(leaf_provider),NONE)
ifneq ($(UNAME_S):$(COB_ENABLE_PORTABLE_STRASSEN),Linux:ON)
$(error An external leaf requires Linux and COB_ENABLE_PORTABLE_STRASSEN=ON)
endif
ifeq ($(filter $(UNAME_M),aarch64 arm64 x86_64 amd64),)
$(error An external leaf requires a native x86-64 or Arm64 target)
endif
ifeq ($(leaf_provider),OPENBLAS)
PKG_CONFIG ?= pkg-config
ifneq ($(shell $(PKG_CONFIG) --exists openblas && echo yes),yes)
$(error The OpenBLAS leaf requires the OpenBLAS development package and its openblas.pc)
endif
SME_DEFINE += -DCOB_ENABLE_OPENBLAS_LEAF=1 $(shell $(PKG_CONFIG) --cflags openblas)
BLAS_LEAF_LIBS := $(shell $(PKG_CONFIG) --libs openblas)
else
ifeq ($(leaf_provider),MKL)
ifeq ($(filter $(UNAME_M),x86_64 amd64),)
$(error The oneMKL leaf requires Linux x86-64)
endif
COB_BLAS_ROOT ?= $(MKLROOT)
leaf_components := mkl_intel_lp64 mkl_sequential mkl_core
SME_DEFINE += -DCOB_ENABLE_MKL_SEQUENTIAL_LEAF=1
else
ifeq ($(filter $(UNAME_M),aarch64 arm64),)
$(error The ArmPL leaf requires Linux Arm64)
endif
COB_BLAS_ROOT ?= $(ARMPL_DIR)
leaf_components := armpl_lp64
SME_DEFINE += -DCOB_ENABLE_ARMPL_SEQUENTIAL_LEAF=1
endif
ifneq ($(strip $(COB_BLAS_ROOT)),)
SME_DEFINE += -I"$(COB_BLAS_ROOT)/include"
leaf_shared = $(firstword $(foreach dir,lib lib/intel64 lib64,$(wildcard $(COB_BLAS_ROOT)/$(dir)/lib$(1).so $(COB_BLAS_ROOT)/$(dir)/lib$(1).so.3)))
else
leaf_shared = $(firstword $(filter-out lib$(1).so lib$(1).so.3,$(foreach suffix,so so.3,$(shell $(CC) -print-file-name=lib$(1).$(suffix)))))
endif
leaf_libraries := $(foreach component,$(leaf_components),$(or $(call leaf_shared,$(component)),$(error Cannot find shared $(component) for $(leaf_provider); set COB_BLAS_ROOT to its installation prefix)))
leaf_rpath := $(foreach directory,$(sort $(dir $(leaf_libraries))),-Wl,-rpath,$(directory))
ifeq ($(leaf_provider),MKL)
BLAS_LEAF_LIBS := $(leaf_rpath) -Wl,--push-state,--no-as-needed $(leaf_libraries) -Wl,--pop-state -lpthread -lm -ldl
else
BLAS_LEAF_LIBS := $(leaf_rpath) $(leaf_libraries) -lm -ldl
endif
endif
ifeq ($(COB_ENABLE_RECURSIVE_STRASSEN),ON)
BLAS_LEAF_TEST := $(BUILD_DIR)/cob_openblas_winograd_test
endif
endif

ifeq ($(COB_ENABLE_HK15),ON)
SME_DEFINE += -DCOB_ENABLE_HK15=1
else ifneq ($(COB_ENABLE_HK15),OFF)
$(error COB_ENABLE_HK15 must be ON or OFF)
endif

ifeq ($(COB_ENABLE_APPLE_MEDIUM_WIDE),ON)
ifneq ($(SME_OBJ),)
APPLE_WIDE_DEFINE := -DCOB_ENABLE_APPLE_MEDIUM_WIDE=1
SME_DEFINE += $(APPLE_WIDE_DEFINE)
APPLE_WIDE_TEST := $(BUILD_DIR)/cob_apple_medium_wide_test
endif
else ifneq ($(COB_ENABLE_APPLE_MEDIUM_WIDE),OFF)
$(error COB_ENABLE_APPLE_MEDIUM_WIDE must be ON or OFF)
endif

ifeq ($(COB_ENABLE_N2_SMALL),ON)
SME_DEFINE += -DCOB_ENABLE_N2_SMALL=1
else ifneq ($(COB_ENABLE_N2_SMALL),OFF)
$(error COB_ENABLE_N2_SMALL must be ON or OFF)
endif

ifeq ($(COB_ENABLE_X86_MEDIUM_ROUTE),ON)
SME_DEFINE += -DCOB_ENABLE_X86_MEDIUM_ROUTE=1
else ifneq ($(COB_ENABLE_X86_MEDIUM_ROUTE),OFF)
$(error COB_ENABLE_X86_MEDIUM_ROUTE must be ON or OFF)
endif

ifeq ($(COB_ENABLE_X86_STREAM_B),ON)
ifeq ($(UNAME_S),Linux)
ifneq ($(filter x86_64 amd64,$(UNAME_M)),)
STREAM_B_OBJS := $(BUILD_DIR)/stream_b_avx2.o $(BUILD_DIR)/stream_b_avx512.o
STREAM_B_TEST := $(BUILD_DIR)/cob_stream_b_test
SME_DEFINE += -DCOB_ENABLE_X86_STREAM_B=1
endif
endif
else ifneq ($(COB_ENABLE_X86_STREAM_B),OFF)
$(error COB_ENABLE_X86_STREAM_B must be ON or OFF)
endif

NATIVE_KERNEL_HEADERS := src/cob_stream_b_dispatch.inc src/cob_cblas_leaf.inc src/cob_openblas_winograd.inc src/cob_hk15.inc third_party/fast-matmul/hk233.inc src/cob_n2_small.inc src/cob_blocked_leaf.inc src/cob_x86_cache_edge.inc src/cob_short_axes.inc src/cob_fused_strassen.inc src/cob_small_x86.inc src/cob_few_row_panel.inc
LIB_OBJ := $(BUILD_DIR)/gemm.o
TEST_BIN := $(BUILD_DIR)/cob_gemm_test
STRASSEN_TEST_OBJ := $(BUILD_DIR)/gemm_strassen_test.o
STRASSEN_TEST_BIN := $(BUILD_DIR)/cob_gemm_strassen_test
STRASSEN_SQUARE_TEST_OBJ := $(BUILD_DIR)/gemm_strassen_square_test.o
STRASSEN_SQUARE_TEST_BIN := $(BUILD_DIR)/cob_gemm_strassen_square_test
BENCH_BIN := $(BUILD_DIR)/cob_gemm_bench
CBLAS_BENCH_BIN := $(BUILD_DIR)/cob_gemm_bench_cblas
FORTRAN_BLAS_BENCH_BIN := $(BUILD_DIR)/cob_gemm_bench_fortran_blas

CBLAS_NAME ?= external_CBLAS
CBLAS_HEADER ?= cblas.h
CBLAS_CFLAGS ?=
CBLAS_LDFLAGS ?=
CBLAS_BENCH_CFLAGS := -DCOB_HAVE_EXTERNAL_CBLAS=1 -DCOB_EXTERNAL_CBLAS_NAME=\"$(CBLAS_NAME)\" -DCOB_EXTERNAL_CBLAS_HEADER=\"$(CBLAS_HEADER)\" $(CBLAS_CFLAGS)
FORTRAN_BLAS_NAME ?= external_sgemm_
FORTRAN_BLAS_LDFLAGS ?=
FORTRAN_BLAS_BENCH_CFLAGS := -DCOB_HAVE_EXTERNAL_FORTRAN_SGEMM=1 -DCOB_EXTERNAL_FORTRAN_SGEMM_NAME=\"$(FORTRAN_BLAS_NAME)\"

.PHONY: all test bench bench-cblas bench-fortran-blas clean FORCE
.PHONY: test-portable-strassen test-recursive-strassen

all: $(TEST_BIN) $(BENCH_BIN)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Track each output separately so reconfiguring a partially built directory
# works even on versions of Make with one-second file timestamps.
SME_CONFIG := $(if $(SME_OBJ),ON:$(COB_SME_CC),OFF) portable:$(COB_ENABLE_PORTABLE_STRASSEN) recursive:$(COB_ENABLE_RECURSIVE_STRASSEN) blas:$(leaf_provider) aligned:$(COB_ENABLE_ALIGNED_WINOGRAD) blas-root:$(COB_BLAS_ROOT) apple-wide:$(COB_ENABLE_APPLE_MEDIUM_WIDE) n2small:$(COB_ENABLE_N2_SMALL) stream-b:$(COB_ENABLE_X86_STREAM_B) x86-medium:$(COB_ENABLE_X86_MEDIUM_ROUTE) hk15:$(COB_ENABLE_HK15)
SME_OUTPUTS := $(LIB_OBJ) $(STRASSEN_TEST_OBJ) $(STRASSEN_SQUARE_TEST_OBJ) $(SME_OBJ) \
               $(TEST_BIN) $(STRASSEN_TEST_BIN) $(STRASSEN_SQUARE_TEST_BIN) \
               $(BENCH_BIN) $(CBLAS_BENCH_BIN) $(FORTRAN_BLAS_BENCH_BIN) $(BLAS_LEAF_TEST)
define cob_track_sme_config
ifneq ($(shell cat "$(1).sme-config" 2>/dev/null),$(SME_CONFIG))
$(1): FORCE
endif
endef
$(foreach target,$(SME_OUTPUTS),$(eval $(call cob_track_sme_config,$(target))))
save_sme_config = @printf '%s\n' '$(SME_CONFIG)' > "$@.sme-config"

$(BUILD_DIR)/strassen_sme_leaf.o: $(SME_SRC) | $(BUILD_DIR)
	$(COB_SME_CC) $(CFLAGS) $(SME_FLAGS) $(APPLE_WIDE_DEFINE) -c $(SME_SRC) -o $@
	$(save_sme_config)

ifneq ($(STREAM_B_OBJS),)
$(BUILD_DIR)/stream_b_avx2.o: src/cob_stream_b_kernel.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -march=x86-64 -mavx2 -mfma -DCOB_STREAM_B_AVX512=0 -c $< -o $@

$(BUILD_DIR)/stream_b_avx512.o: src/cob_stream_b_kernel.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -march=x86-64 -mavx512f -mfma -DCOB_STREAM_B_AVX512=1 -c $< -o $@

$(STREAM_B_TEST): tests/test_stream_b.c src/gemm.c include/cob_gemm.h $(NATIVE_KERNEL_HEADERS) $(STREAM_B_OBJS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) -DCOB_ENABLE_X86_STREAM_B=1 $< $(STREAM_B_OBJS) -lm -o $@
endif

$(LIB_OBJ): src/gemm.c include/cob_gemm.h $(NATIVE_KERNEL_HEADERS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(SME_DEFINE) -c src/gemm.c -o $@
	$(save_sme_config)

$(TEST_BIN): tests/test_gemm.c include/cob_gemm.h $(LIB_OBJ) $(SME_OBJ) $(STREAM_B_OBJS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) tests/test_gemm.c $(LIB_OBJ) $(SME_OBJ) $(STREAM_B_OBJS) $(BLAS_LEAF_LIBS) -o $@
	$(save_sme_config)

$(STRASSEN_TEST_OBJ): src/gemm.c include/cob_gemm.h $(NATIVE_KERNEL_HEADERS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(SME_DEFINE) -DCOB_SGEMM_STRASSEN1_MIN_DIM=1024 -c src/gemm.c -o $@
	$(save_sme_config)

$(STRASSEN_TEST_BIN): tests/test_gemm.c include/cob_gemm.h $(STRASSEN_TEST_OBJ) $(SME_OBJ) $(STREAM_B_OBJS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) tests/test_gemm.c $(STRASSEN_TEST_OBJ) $(SME_OBJ) $(STREAM_B_OBJS) $(BLAS_LEAF_LIBS) -o $@
	$(save_sme_config)

$(STRASSEN_SQUARE_TEST_OBJ): src/gemm.c include/cob_gemm.h $(NATIVE_KERNEL_HEADERS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(SME_DEFINE) -DCOB_SGEMM_STRASSEN1_MIN_DIM=8192 -DCOB_SGEMM_STRASSEN1_SQUARE_MIN_DIM=128 -c src/gemm.c -o $@
	$(save_sme_config)

$(STRASSEN_SQUARE_TEST_BIN): tests/test_gemm.c include/cob_gemm.h $(STRASSEN_SQUARE_TEST_OBJ) $(SME_OBJ) $(STREAM_B_OBJS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) tests/test_gemm.c $(STRASSEN_SQUARE_TEST_OBJ) $(SME_OBJ) $(STREAM_B_OBJS) $(BLAS_LEAF_LIBS) -o $@
	$(save_sme_config)

$(BENCH_BIN): bench/bench_gemm.c include/cob_gemm.h $(LIB_OBJ) $(SME_OBJ) $(STREAM_B_OBJS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(BENCH_CFLAGS) bench/bench_gemm.c $(LIB_OBJ) $(SME_OBJ) $(STREAM_B_OBJS) $(BENCH_LDFLAGS) $(BLAS_LEAF_LIBS) -o $@
	$(save_sme_config)

$(CBLAS_BENCH_BIN): bench/bench_gemm.c include/cob_gemm.h $(LIB_OBJ) $(SME_OBJ) $(STREAM_B_OBJS) FORCE | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(CBLAS_BENCH_CFLAGS) bench/bench_gemm.c $(LIB_OBJ) $(SME_OBJ) $(STREAM_B_OBJS) $(CBLAS_LDFLAGS) $(BLAS_LEAF_LIBS) -o $@
	$(save_sme_config)

$(FORTRAN_BLAS_BENCH_BIN): bench/bench_gemm.c include/cob_gemm.h $(LIB_OBJ) $(SME_OBJ) $(STREAM_B_OBJS) FORCE | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(FORTRAN_BLAS_BENCH_CFLAGS) bench/bench_gemm.c $(LIB_OBJ) $(SME_OBJ) $(STREAM_B_OBJS) $(FORTRAN_BLAS_LDFLAGS) $(BLAS_LEAF_LIBS) -o $@
	$(save_sme_config)

ifneq ($(BLAS_LEAF_TEST),)
$(BLAS_LEAF_TEST): tests/test_openblas_winograd.c src/gemm.c include/cob_gemm.h $(NATIVE_KERNEL_HEADERS) $(STREAM_B_OBJS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(SME_DEFINE) $< $(STREAM_B_OBJS) $(BLAS_LEAF_LIBS) -lm -o $@
	$(save_sme_config)
endif

ifneq ($(APPLE_WIDE_TEST),)
$(APPLE_WIDE_TEST): tests/test_apple_medium_wide.c $(LIB_OBJ) $(SME_OBJ) $(STREAM_B_OBJS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $< $(LIB_OBJ) $(SME_OBJ) $(STREAM_B_OBJS) $(BLAS_LEAF_LIBS) -o $@
endif

test: $(TEST_BIN) $(STRASSEN_TEST_BIN) $(STRASSEN_SQUARE_TEST_BIN) $(BLAS_LEAF_TEST) $(STREAM_B_TEST) $(APPLE_WIDE_TEST)
	$(TEST_BIN)
	$(STRASSEN_TEST_BIN)
	$(STRASSEN_SQUARE_TEST_BIN)
	$(BLAS_LEAF_TEST)
	$(STREAM_B_TEST)
	$(APPLE_WIDE_TEST)

test-portable-strassen: $(BUILD_DIR)/cob_portable_strassen_test
	$<

test-recursive-strassen: $(BUILD_DIR)/cob_recursive_strassen_test
	$<

$(BUILD_DIR)/cob_recursive_strassen_test: RECURSIVE_TEST_FLAGS := -DCOB_ENABLE_RECURSIVE_STRASSEN=1 -DCOB_RECURSIVE_MIN_DIM=128 -DCOB_RECURSIVE_MAX_DIM=256
$(BUILD_DIR)/cob_recursive_strassen_test $(BUILD_DIR)/cob_portable_strassen_test: tests/test_portable_strassen.c src/gemm.c include/cob_gemm.h $(NATIVE_KERNEL_HEADERS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(RECURSIVE_TEST_FLAGS) -DCOB_ENABLE_PORTABLE_STRASSEN=1 -DCOB_PORTABLE_STRASSEN_MIN_DIM=128 -DCOB_RECTANGULAR_MIN_HARMONIC_DIM=192 \
	    -DCOB_DISABLE_APPLE_AMX=1 -DCOB_DISABLE_APPLE_SME=1 $< -lm -o $@

bench: $(BENCH_BIN)
	$(BENCH_BIN)

bench-cblas: $(CBLAS_BENCH_BIN)
	$(CBLAS_BENCH_BIN)

bench-fortran-blas: $(FORTRAN_BLAS_BENCH_BIN)
	$(FORTRAN_BLAS_BENCH_BIN)

clean:
	rm -rf $(BUILD_DIR)

FORCE:
