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
.PHONY: test-portable-strassen

all: $(TEST_BIN) $(BENCH_BIN)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Track each output separately so reconfiguring a partially built directory
# works even on versions of Make with one-second file timestamps.
SME_CONFIG := $(if $(SME_OBJ),ON:$(COB_SME_CC),OFF) portable:$(COB_ENABLE_PORTABLE_STRASSEN)
SME_OUTPUTS := $(LIB_OBJ) $(STRASSEN_TEST_OBJ) $(STRASSEN_SQUARE_TEST_OBJ) $(SME_OBJ) \
               $(TEST_BIN) $(STRASSEN_TEST_BIN) $(STRASSEN_SQUARE_TEST_BIN) \
               $(BENCH_BIN) $(CBLAS_BENCH_BIN) $(FORTRAN_BLAS_BENCH_BIN)
define cob_track_sme_config
ifneq ($(shell cat "$(1).sme-config" 2>/dev/null),$(SME_CONFIG))
$(1): FORCE
endif
endef
$(foreach target,$(SME_OUTPUTS),$(eval $(call cob_track_sme_config,$(target))))
save_sme_config = @printf '%s\n' '$(SME_CONFIG)' > "$@.sme-config"

$(BUILD_DIR)/strassen_sme_leaf.o: $(SME_SRC) | $(BUILD_DIR)
	$(COB_SME_CC) $(CFLAGS) $(SME_FLAGS) -c $(SME_SRC) -o $@
	$(save_sme_config)

$(LIB_OBJ): src/gemm.c include/cob_gemm.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(SME_DEFINE) -c src/gemm.c -o $@
	$(save_sme_config)

$(TEST_BIN): tests/test_gemm.c include/cob_gemm.h $(LIB_OBJ) $(SME_OBJ) | $(BUILD_DIR)
	$(CC) $(CFLAGS) tests/test_gemm.c $(LIB_OBJ) $(SME_OBJ) -o $@
	$(save_sme_config)

$(STRASSEN_TEST_OBJ): src/gemm.c include/cob_gemm.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(SME_DEFINE) -DCOB_SGEMM_STRASSEN1_MIN_DIM=1024 -c src/gemm.c -o $@
	$(save_sme_config)

$(STRASSEN_TEST_BIN): tests/test_gemm.c include/cob_gemm.h $(STRASSEN_TEST_OBJ) $(SME_OBJ) | $(BUILD_DIR)
	$(CC) $(CFLAGS) tests/test_gemm.c $(STRASSEN_TEST_OBJ) $(SME_OBJ) -o $@
	$(save_sme_config)

$(STRASSEN_SQUARE_TEST_OBJ): src/gemm.c include/cob_gemm.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(SME_DEFINE) -DCOB_SGEMM_STRASSEN1_MIN_DIM=8192 -DCOB_SGEMM_STRASSEN1_SQUARE_MIN_DIM=128 -c src/gemm.c -o $@
	$(save_sme_config)

$(STRASSEN_SQUARE_TEST_BIN): tests/test_gemm.c include/cob_gemm.h $(STRASSEN_SQUARE_TEST_OBJ) $(SME_OBJ) | $(BUILD_DIR)
	$(CC) $(CFLAGS) tests/test_gemm.c $(STRASSEN_SQUARE_TEST_OBJ) $(SME_OBJ) -o $@
	$(save_sme_config)

$(BENCH_BIN): bench/bench_gemm.c include/cob_gemm.h $(LIB_OBJ) $(SME_OBJ) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(BENCH_CFLAGS) bench/bench_gemm.c $(LIB_OBJ) $(SME_OBJ) $(BENCH_LDFLAGS) -o $@
	$(save_sme_config)

$(CBLAS_BENCH_BIN): bench/bench_gemm.c include/cob_gemm.h $(LIB_OBJ) $(SME_OBJ) FORCE | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(CBLAS_BENCH_CFLAGS) bench/bench_gemm.c $(LIB_OBJ) $(SME_OBJ) $(CBLAS_LDFLAGS) -o $@
	$(save_sme_config)

$(FORTRAN_BLAS_BENCH_BIN): bench/bench_gemm.c include/cob_gemm.h $(LIB_OBJ) $(SME_OBJ) FORCE | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(FORTRAN_BLAS_BENCH_CFLAGS) bench/bench_gemm.c $(LIB_OBJ) $(SME_OBJ) $(FORTRAN_BLAS_LDFLAGS) -o $@
	$(save_sme_config)

test: $(TEST_BIN) $(STRASSEN_TEST_BIN) $(STRASSEN_SQUARE_TEST_BIN)
	$(TEST_BIN)
	$(STRASSEN_TEST_BIN)
	$(STRASSEN_SQUARE_TEST_BIN)

test-portable-strassen: $(BUILD_DIR)/cob_portable_strassen_test
	$<

$(BUILD_DIR)/cob_portable_strassen_test: tests/test_portable_strassen.c src/gemm.c include/cob_gemm.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -DCOB_ENABLE_PORTABLE_STRASSEN=1 -DCOB_PORTABLE_STRASSEN_MIN_DIM=128 \
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
