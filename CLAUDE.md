# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repo benchmarks CUDA FP128 (128-bit floating-point) kernels built with Kokkos, comparing device results against host `quadmath.h` references to verify accuracy. It produces two executables: one for 39 real math operations and one for 24 complex math operations.

## Build

The build requires Kokkos ≥5.1 with CUDA backend targeting Blackwell (sm_100), GCC 13.3.0, CMake 3.28.3, and CUDA 12.9.1. On Argonne systems, load modules first:

```bash
source scripts/prepare.sh
```

Then build (clones and installs Kokkos 5.1.0, then builds the demo):

```bash
source scripts/build_with_kokkos.sh <install-dir>
```

This outputs `build/kokkos_ep_demo` and `build/kokkos_ep_demo_complex`, and generates `setup.sh` with the necessary environment variables (`KOKKOS_HOME`, `CMAKE_PREFIX_PATH`, `LD_LIBRARY_PATH`, etc.).

If Kokkos is already installed, build directly with CMake:

```bash
cmake -B build -DCMAKE_PREFIX_PATH=<kokkos-install-dir>
cmake --build build -j$(nproc)
```

## Running Benchmarks

```bash
# All real operations (single table output)
./build/kokkos_ep_demo --batch 500000 --repeats 5

# Single real operation
./build/kokkos_ep_demo --op sin --batch 1000000 --repeats 5

# All complex operations
./build/kokkos_ep_demo_complex --batch 500000 --repeats 5

# Single complex operation
./build/kokkos_ep_demo_complex --op exp --batch 1000000 --repeats 5
```

Arguments: `--op <name>`, `--batch N` (default: 1,000,000), `--repeats N` (default: 5), `--seed N` (default: 12345).

Scripts for batch runs: `scripts/run_all_ops.sh` and `scripts/run_all_complex_ops.sh`.

## Architecture

### Three-Layer Design

```
Demo Executables (src/demo_real.cpp, src/demo_complex.cpp)
    └─ CUDA FP128 Wrapper Layer
       (third_party/include/NVIDIA_emulated_quad/quad_math.hpp, quad_complex.hpp)
       └─ Kokkos Runtime + CUDA Backend
```

### FP128 Wrapper (`third_party/include/NVIDIA_emulated_quad/`)

`quad_math.hpp` defines `fp128_t` (a wrapper around `__float128`/`__nv_fp128_base`) with full operator overloading. The critical design is the dual-path compilation:

```cpp
#ifdef __CUDA_ARCH__
    // Device: calls NVIDIA __nv_fp128_* functions
#else
    // Host: uses native C++ operators / quadmath.h
#endif
```

All functions are marked `KOKKOS_INLINE_FUNCTION` and have explicit (non-defaulted) copy constructors/assignments because NVCC requires explicit function bodies for device code generation.

`quad_complex.hpp` defines a minimal `quad_complex` struct (not `Kokkos::complex<fp128_t>`) because the Kokkos template causes device code generation issues. It stores `fp128_t` real and imaginary parts and dispatches through `quad::cuda_fp128::*` functions.

### Demo Executables (`src/`)

Each demo follows the same pattern:
1. Parse args → generate operation-specific random inputs on host
2. Compute `quadmath.h` host reference (ground truth)
3. Deep copy inputs to Kokkos device Views
4. Run 2 warmup + N timed `Kokkos::parallel_for` launches with `Kokkos::fence()` after each
5. Deep copy results back, compute per-element relative error → convert to "digits of accuracy" (`-log10(rel_error)`, clamped to [0, 33])
6. Print formatted table: FP128 vs FP64 timing and accuracy statistics

Timing includes `Kokkos::fence()` overhead (synchronization cost), not raw kernel time.

Accuracy is reported as precise decimal digits: 33 = perfect FP128 match, 0 = completely wrong.

### Platform Constraint

`libquadmath` (host reference) is x86_64 only. CMake enforces this — the project will not build on ARM or other platforms.
