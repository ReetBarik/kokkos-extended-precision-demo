# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the `ddfun` branch. It benchmarks Kokkos double-double (DD) arithmetic kernels, comparing device results against host `quadmath.h` references (~34 digits) to measure accuracy. DD arithmetic provides ~30-31 decimal digits of precision using two `double` values per number. It produces two executables: one for 39 real math operations and one for 24 complex math operations.

## Build

The build requires Kokkos ≥5.1 with CUDA backend, GCC 13.3.0, CMake 3.28.3, and CUDA 12.9.1. On Argonne systems, load modules first:

```bash
source scripts/prepare.sh
```

Then build (clones and installs Kokkos 5.1.0, then builds the demo):

```bash
source scripts/build_with_kokkos.sh <install-dir>
```

This outputs `build/kokkos_dd_demo` and `build/kokkos_dd_demo_complex`, and generates `setup.sh` with the necessary environment variables (`KOKKOS_HOME`, `CMAKE_PREFIX_PATH`, `LD_LIBRARY_PATH`, etc.).

If Kokkos is already installed, build directly with CMake:

```bash
cmake -B build -DCMAKE_PREFIX_PATH=<kokkos-install-dir>
cmake --build build -j$(nproc)
```

## Running Benchmarks

```bash
# All real operations (single table output)
./build/kokkos_dd_demo --batch 500000 --repeats 5

# Single real operation
./build/kokkos_dd_demo --op sin --batch 1000000 --repeats 5

# All complex operations
./build/kokkos_dd_demo_complex --batch 500000 --repeats 5

# Single complex operation
./build/kokkos_dd_demo_complex --op exp --batch 1000000 --repeats 5
```

Arguments: `--op <name>`, `--batch N` (default: 1,000,000), `--repeats N` (default: 5), `--seed N` (default: 12345).

Scripts for batch runs: `scripts/run_all_dd_ops.sh` and `scripts/run_all_dd_complex_ops.sh`.

## Architecture

### Three-Layer Design

```
Demo Executables (src/demo_dd_real.cpp, src/demo_dd_complex.cpp)
    └─ Double-Double Math Library
       (third_party/include/dd_math.hpp, dd_complex.hpp)
       └─ Kokkos Runtime + CUDA Backend
```

### Double-Double Math Library (`third_party/include/`)

`dd_math.hpp` defines `ddouble { double hi; double lo; }` in `namespace quad::ddfun`. The `hi` component holds the primary value; `lo` holds the rounding error from `hi`. Key algorithms:

- **TwoSum (Knuth)**: basis of `ddadd` / `ddsub` — error-free summation
- **TwoProduct (Dekker splitting)**: basis of `ddmul` / `dddiv` — no FMA required
- **Constants**: constructed from IEEE 754 bit patterns via `make_dd()` for both host (`std::memcpy`) and device (`__longlong_as_double`)

All functions are `KOKKOS_INLINE_FUNCTION`, ported from DDFUN (David H. Bailey, Lawrence Berkeley National Lab).

`dd_complex.hpp` defines `ddcomplex { ddouble re; ddouble im; }` with full complex math via `quad::ddfun::` free functions.

### Demo Executables (`src/`)

Each demo follows the same pattern:
1. Parse args → generate operation-specific random inputs on host
2. Compute `quadmath.h` host reference (ground truth, ~34 digits)
3. Deep copy inputs to Kokkos device Views as `ddouble`
4. Run 2 warmup + N timed `Kokkos::parallel_for` launches with `Kokkos::fence()` after each
5. Deep copy results back, compute per-element relative error → convert to "digits of accuracy" (`-log10(rel_error)`, clamped to [0, 31])
6. Print formatted table: DD vs FP64 timing and accuracy statistics

Timing includes `Kokkos::fence()` overhead (synchronization cost), not raw kernel time.

Accuracy is reported as precise decimal digits: 31 = perfect DD match (~30-31 digits max), 0 = completely wrong.

### Namespace

All DD types and functions live in `namespace quad::ddfun` (symmetric with `quad::cuda_fp128::` on the `main` branch).

### Platform Constraint

`libquadmath` (host reference) is x86_64 only. CMake enforces this — the project will not build on ARM or other platforms.
