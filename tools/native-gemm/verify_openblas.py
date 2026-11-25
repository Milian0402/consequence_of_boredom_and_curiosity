"""Check the Linux Make/CMake OpenBLAS option against the frozen benchmark."""
import os
from pathlib import Path
import platform
import re
import shlex
import subprocess
import sys
import tempfile

PIN = "29adce432ce47e1ee5a9ce53b932f5760af14ced"
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CC = shlex.split(os.environ.get("CC", "cc"))
EXPORTED = (
    "sgemm_rowmajor", "sgemm_ref_rowmajor", "sgemm_pack_a", "sgemm_free_packed_a",
    "sgemm_pack_b", "sgemm_free_packed_b", "sgemm_rowmajor_packed_b",
    "sgemm_rowmajor_packed_ab",
)


def output(args):
    return subprocess.check_output(args, cwd=ROOT, text=True)


def run(args, **kwargs):
    subprocess.run(args, cwd=ROOT, check=True, **kwargs)


def verify(stage):
    arch = platform.machine()
    if platform.system() != "Linux" or arch not in ("x86_64", "aarch64"):
        raise SystemExit("requires a Linux x86-64 or Arm64 runner")
    # Ubuntu provides the threaded OpenBLAS metadata in this multiarch subdirectory.
    multiarch = output(["gcc", "-print-multiarch"]).strip()
    pc = Path("/usr/lib") / multiarch / "openblas-pthread/pkgconfig"
    if pc.is_dir():
        os.environ["PKG_CONFIG_PATH"] = os.pathsep.join([str(pc), os.getenv("PKG_CONFIG_PATH", "")])
    cblas = shlex.split(output(["pkg-config", "--cflags", "openblas"]))
    libs = shlex.split(output(["pkg-config", "--libs", "openblas"]))
    small = ["-DCOB_PORTABLE_STRASSEN_MIN_DIM=128", "-DCOB_RECTANGULAR_MIN_HARMONIC_DIM=192", "-DCOB_ENABLE_RECTANGULAR_STRASSEN=1"]
    native = ["-DCOB_USE_NEON=1"] if arch == "aarch64" else []
    flags = ["-std=c11", "-O2", "-Iinclude", "-Isrc", "-DCOB_ENABLE_PORTABLE_STRASSEN=1", *small, *native]
    leaf = "-DCOB_ENABLE_OPENBLAS_LEAF=1"
    previous, generator, measured = (stage / name for name in ("previous.c", "generator.py", "measured.c"))
    previous.write_text(output(["git", "show", f"{PIN}:src/gemm.c"]))
    generator.write_text(output(["git", "show", f"{PIN}:tools/native-gemm/openblas_leaf.py"]))
    measured.write_text(output([sys.executable, generator, previous]))

    def preprocess_leaf(source, options):
        # The public dispatch policy can evolve independently of each leaf.
        program = output([*CC, *flags, *options, "-E", "-P", source])
        match = re.search(r"static\s+int\s+cob_strassen_leaf\s*\([^;{}]*\)\s*\{", program)
        if not match:
            raise RuntimeError("portable Strassen leaf is missing")
        depth = 1
        for pos in range(match.end(), len(program)):
            char = program[pos]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return program[match.start():pos+1].split()
        raise RuntimeError("unterminated portable Strassen leaf")

    for recursive in ([], ["-DCOB_ENABLE_RECURSIVE_STRASSEN=1"]):
        if preprocess_leaf("src/gemm.c", recursive) != preprocess_leaf(previous, recursive):
            raise RuntimeError("the disabled leaf differs from the pinned native program")
        if preprocess_leaf("src/gemm.c", [*recursive, *cblas, leaf]) != preprocess_leaf(measured, [*recursive, *cblas]):
            raise RuntimeError("the enabled leaf differs from the pinned benchmark generator")

    base = stage / "baseline.o"
    renames = [f"-Dcob_{name}=baseline_{name}" for name in EXPORTED]
    run([*CC, *flags, *renames, "-c", previous, "-o", base])
    for mode in ("OFF", "ON"):
        make_dir = stage / f"make-{mode}"
        cmake_dir = stage / f"cmake-{mode}"
        options = ["COB_ENABLE_PORTABLE_STRASSEN=ON", f"COB_ENABLE_OPENBLAS_LEAF={mode}"]
        cflags = " ".join(["-O2", *small])
        run(["make", *options, f"BUILD_DIR={make_dir}", make_dir / "cob_gemm_bench"],
            env={**os.environ, "CFLAGS": cflags})
        run(["cmake", "-S", ".", "-B", cmake_dir, "-DCOB_BUILD_TESTS=OFF", "-DCOB_BUILD_BENCH=ON",
             f"-DCMAKE_C_FLAGS={cflags}", *["-D" + option for option in options]])
        run(["cmake", "--build", cmake_dir, "--target", "cob_gemm_bench"])
        for obj in (make_dir / "gemm.o", cmake_dir / "libcob_matmul.a"):
            executable = stage / "smoke"
            run([*CC, "-O2", "-Iinclude", *cblas, HERE / "openblas_leaf_smoke.c",
                 base, obj, "-Wl,--wrap=cblas_sgemm", *libs, "-lm", "-o", executable])
            run([executable, mode])


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="cob-openblas-") as scratch:
        verify(Path(scratch))
