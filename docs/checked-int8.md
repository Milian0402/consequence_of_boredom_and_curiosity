# Checked integer multiplication for Linux

C programs can pass ordinary row-major FP32 matrices to `exact_int8_checked_sgemm` and supply their usual SGEMM function as a fallback. The optional shared library checks the live values on each eligible call. On a supported CPU it can multiply exact signed bytes and accumulate in signed 32-bit integers. It returns the result in the caller's FP32 output. On other valid inputs the callback computes the complete product.

## Build and install

On Linux x86-64 or Arm64, use a C++17 compiler and set `DNNL_PREFIX` to the installed oneDNN SDK prefix. The benchmark used oneDNN source tag `v3.14.2`, whose headers and runtime report **3.14.1**. Build oneDNN for a sequential CPU runtime with GPU support disabled. [Pinned oneDNN source](https://github.com/uxlfoundation/oneDNN/tree/4db284662a37cf46401655b5ee99391d9de52c00) · [Benchmark runtime details](../tools/world-rank/precision/backend/README.md)

```sh
cmake -S . -B build-checked -DCMAKE_BUILD_TYPE=Release \
  -DCOB_BUILD_CHECKED_INT8=ON -DCOB_DNNL_INCLUDE_DIR="$DNNL_PREFIX/include" \
  -DCMAKE_INSTALL_LIBDIR=lib -DCOB_BUILD_BENCH=OFF
cmake --build build-checked
ctest --test-dir build-checked -R cob_checked_int8_test --output-on-failure
cmake --install build-checked --prefix "$PWD/install-checked"
```

The target `cob_checked_int8` installs `libcob_checked_int8.so` and `checked-int8.h`. The usual `cob_matmul` target supplies the FP32 callback in the example. The checked adapter uses the C++ runtime, threads and the Linux dynamic loader. The C caller needs only the two installed COB headers. Turning the option off leaves a C-only build.

## Call from C

Save this example as `example.c`. Each of its 19 × 27 outputs should equal 2 × 3 × 256 = 1,536.

```c
#include <checked-int8.h>
#include <cob_gemm.h>
#include <stdio.h>

int main(void) {
    enum { rows = 19, cols = 27, depth = 256 };
    float a[rows * depth], b[depth * cols], c[rows * cols];
    for (int i = 0; i < rows * depth; ++i)
        a[i] = 2.0f;
    for (int i = 0; i < depth * cols; ++i)
        b[i] = 3.0f;

    int route = exact_int8_checked_sgemm(rows, cols, depth, a, depth, b, cols,
                                         c, cols, cob_sgemm_rowmajor);
    printf("C[0] = %.0f, route = %d (%s)\n", c[0], route,
           exact_int8_checked_last_reason());
    for (int i = 0; i < rows * cols; ++i)
        if (c[i] != 1536.0f)
            return 1;
    return route < 1;
}
```

```sh
prefix="$PWD/install-checked"
cc -std=c11 example.c -I"$prefix/include" -L"$prefix/lib" \
  -Wl,-rpath,"$prefix/lib" -lcob_checked_int8 -lcob_matmul -lm -o example
export EXACT_INT8_DNNL_PATH="$DNNL_PREFIX/lib/libdnnl.so.3"
export ONEDNN_PRIMITIVE_CACHE_CAPACITY=0
./example
```

Set the environment before starting the process. The native route requires **oneDNN 3.14.1**, sequential CPU runtime, GPU runtime disabled and primitive cache capacity zero. If `EXACT_INT8_DNNL_PATH` is unset, the loader searches for `libdnnl.so.3`. On x86-64 the adapter requires AMX INT8 and kernel permission to use tile state. On Arm64 it requires SVE I8MM. The adapter also checks the selected oneDNN implementation for the actual geometry. A machine can satisfy the ISA checks and use the fallback for a particular shape. To run the example through its callback explicitly, set `EXACT_INT8_NATIVE_ENABLED=0` before launching it.

## Contract

Use valid FP32 views for the overwrite product `C = A × B`. Strides are in float elements. C must be disjoint from A and B. Always pass a callback with the signature in `checked-int8.h`. It must handle the same row-major product and strides. Use a single-threaded callback when single-thread execution is required.

The integer route requires 129 ≤ K ≤ 4,096, finite integer values in [−128,127], and K × max|A| × max|B| ≤ 2²⁴. This bounds every true partial sum and guarantees exact FP32 representation of the final integer. Signed zeros compare equal numerically. A native call includes validation, packing, creating and executing a oneDNN primitive, and copying its private result back to C. Per-thread engine state may persist between calls.

| Return | Outcome |
|---|---|
| 1 | The checked integer route completed the product. |
| 2 | A depth, view or input-domain check delegated to the callback. |
| 3 | Hardware, backend, workspace or execution availability delegated to the callback. |
| −1 | Missing callback. The entry returned before computing the product. |

The thread-local `exact_int8_checked_last_reason()` gives the reason for the most recent call on the current thread. A return of 2 or 3 means the callback already ran. Its output follows the chosen callback's ordinary FP32 numerical behavior. The benchmark used independently selected callbacks for its Linux comparisons. This example uses COB's installed default FP32 implementation. [Benchmarked source and protocol](../tools/world-rank/exact/s8/protocol.md)
