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

COB_ENABLE_PORTABLE_STRASSEN ?= OFF
COB_ENABLE_RECURSIVE_STRASSEN ?= OFF
COB_ENABLE_OPENBLAS_LEAF ?= OFF

ifeq ($(COB_ENABLE_PORTABLE_STRASSEN),ON)
NATIVE_CPPFLAGS += -DCOB_ENABLE_PORTABLE_STRASSEN=1
else ifneq ($(COB_ENABLE_PORTABLE_STRASSEN),OFF)
$(error COB_ENABLE_PORTABLE_STRASSEN must be ON or OFF)
endif

ifeq ($(COB_ENABLE_RECURSIVE_STRASSEN),ON)
NATIVE_CPPFLAGS += -DCOB_ENABLE_RECURSIVE_STRASSEN=1
else ifneq ($(COB_ENABLE_RECURSIVE_STRASSEN),OFF)
$(error COB_ENABLE_RECURSIVE_STRASSEN must be ON or OFF)
endif

ifeq ($(COB_ENABLE_OPENBLAS_LEAF),ON)
ifneq ($(UNAME_S):$(COB_ENABLE_PORTABLE_STRASSEN),Linux:ON)
$(error COB_ENABLE_OPENBLAS_LEAF=ON requires Linux and COB_ENABLE_PORTABLE_STRASSEN=ON)
endif
ifeq ($(filter $(UNAME_M),aarch64 arm64 x86_64 amd64),)
$(error COB_ENABLE_OPENBLAS_LEAF=ON requires a native x86-64 or Arm64 target)
endif
PKG_CONFIG ?= pkg-config
ifneq ($(shell $(PKG_CONFIG) --exists openblas && echo yes),yes)
$(error COB_ENABLE_OPENBLAS_LEAF=ON requires the OpenBLAS development package and its openblas.pc)
endif
NATIVE_CPPFLAGS += -DCOB_ENABLE_OPENBLAS_LEAF=1 $(shell $(PKG_CONFIG) --cflags openblas)
OPENBLAS_LEAF_LIBS := $(shell $(PKG_CONFIG) --libs openblas)
else ifneq ($(COB_ENABLE_OPENBLAS_LEAF),OFF)
$(error COB_ENABLE_OPENBLAS_LEAF must be ON or OFF)
endif

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

all: $(TEST_BIN) $(BENCH_BIN)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Rebuild each affected object and executable after a backend option
# changes, including partial builds and toolchains with coarse timestamps.
NATIVE_CONFIG := portable:$(COB_ENABLE_PORTABLE_STRASSEN) recursive:$(COB_ENABLE_RECURSIVE_STRASSEN) openblas:$(COB_ENABLE_OPENBLAS_LEAF)
define cob_track_native_config
ifneq ($(shell cat "$(1).native-config" 2>/dev/null),$(NATIVE_CONFIG))
$(1): FORCE
endif
endef
NATIVE_OUTPUTS := $(LIB_OBJ) $(STRASSEN_TEST_OBJ) $(STRASSEN_SQUARE_TEST_OBJ) \
    $(TEST_BIN) $(STRASSEN_TEST_BIN) $(STRASSEN_SQUARE_TEST_BIN) $(BENCH_BIN) $(CBLAS_BENCH_BIN) $(FORTRAN_BLAS_BENCH_BIN)
$(foreach target,$(NATIVE_OUTPUTS),$(eval $(call cob_track_native_config,$(target))))
save_native_config = @printf '%s\n' '$(NATIVE_CONFIG)' > "$@.native-config"

$(LIB_OBJ): src/gemm.c src/cob_openblas_winograd.inc include/cob_gemm.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(NATIVE_CPPFLAGS) -c src/gemm.c -o $@
	$(save_native_config)

$(TEST_BIN): tests/test_gemm.c include/cob_gemm.h $(LIB_OBJ) | $(BUILD_DIR)
	$(CC) $(CFLAGS) tests/test_gemm.c $(LIB_OBJ) $(OPENBLAS_LEAF_LIBS) -o $@
	$(save_native_config)

$(STRASSEN_TEST_OBJ): src/gemm.c src/cob_openblas_winograd.inc include/cob_gemm.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(NATIVE_CPPFLAGS) -DCOB_SGEMM_STRASSEN1_MIN_DIM=1024 -c src/gemm.c -o $@
	$(save_native_config)

$(STRASSEN_TEST_BIN): tests/test_gemm.c include/cob_gemm.h $(STRASSEN_TEST_OBJ) | $(BUILD_DIR)
	$(CC) $(CFLAGS) tests/test_gemm.c $(STRASSEN_TEST_OBJ) $(OPENBLAS_LEAF_LIBS) -o $@
	$(save_native_config)

$(STRASSEN_SQUARE_TEST_OBJ): src/gemm.c src/cob_openblas_winograd.inc include/cob_gemm.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(NATIVE_CPPFLAGS) -DCOB_SGEMM_STRASSEN1_MIN_DIM=8192 -DCOB_SGEMM_STRASSEN1_SQUARE_MIN_DIM=128 -c src/gemm.c -o $@
	$(save_native_config)

$(STRASSEN_SQUARE_TEST_BIN): tests/test_gemm.c include/cob_gemm.h $(STRASSEN_SQUARE_TEST_OBJ) | $(BUILD_DIR)
	$(CC) $(CFLAGS) tests/test_gemm.c $(STRASSEN_SQUARE_TEST_OBJ) $(OPENBLAS_LEAF_LIBS) -o $@
	$(save_native_config)

$(BENCH_BIN): bench/bench_gemm.c include/cob_gemm.h $(LIB_OBJ) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(BENCH_CFLAGS) bench/bench_gemm.c $(LIB_OBJ) $(BENCH_LDFLAGS) $(OPENBLAS_LEAF_LIBS) -o $@
	$(save_native_config)

$(CBLAS_BENCH_BIN): bench/bench_gemm.c include/cob_gemm.h $(LIB_OBJ) FORCE | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(CBLAS_BENCH_CFLAGS) bench/bench_gemm.c $(LIB_OBJ) $(CBLAS_LDFLAGS) $(OPENBLAS_LEAF_LIBS) -o $@
	$(save_native_config)

$(FORTRAN_BLAS_BENCH_BIN): bench/bench_gemm.c include/cob_gemm.h $(LIB_OBJ) FORCE | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(FORTRAN_BLAS_BENCH_CFLAGS) bench/bench_gemm.c $(LIB_OBJ) $(FORTRAN_BLAS_LDFLAGS) $(OPENBLAS_LEAF_LIBS) -o $@
	$(save_native_config)

test: $(TEST_BIN) $(STRASSEN_TEST_BIN) $(STRASSEN_SQUARE_TEST_BIN)
	$(TEST_BIN)
	$(STRASSEN_TEST_BIN)
	$(STRASSEN_SQUARE_TEST_BIN)

bench: $(BENCH_BIN)
	$(BENCH_BIN)

bench-cblas: $(CBLAS_BENCH_BIN)
	$(CBLAS_BENCH_BIN)

bench-fortran-blas: $(FORTRAN_BLAS_BENCH_BIN)
	$(FORTRAN_BLAS_BENCH_BIN)

clean:
	rm -rf $(BUILD_DIR)

FORCE:
