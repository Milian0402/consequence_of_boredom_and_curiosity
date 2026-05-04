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

LIB_OBJ := $(BUILD_DIR)/gemm.o
TEST_BIN := $(BUILD_DIR)/cob_gemm_test
BENCH_BIN := $(BUILD_DIR)/cob_gemm_bench

.PHONY: all test bench clean

all: $(TEST_BIN) $(BENCH_BIN)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(LIB_OBJ): src/gemm.c include/cob_gemm.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c src/gemm.c -o $@

$(TEST_BIN): tests/test_gemm.c include/cob_gemm.h $(LIB_OBJ) | $(BUILD_DIR)
	$(CC) $(CFLAGS) tests/test_gemm.c $(LIB_OBJ) -o $@

$(BENCH_BIN): bench/bench_gemm.c include/cob_gemm.h $(LIB_OBJ) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(BENCH_CFLAGS) bench/bench_gemm.c $(LIB_OBJ) $(BENCH_LDFLAGS) -o $@

test: $(TEST_BIN)
	$(TEST_BIN)

bench: $(BENCH_BIN)
	$(BENCH_BIN)

clean:
	rm -rf $(BUILD_DIR)
