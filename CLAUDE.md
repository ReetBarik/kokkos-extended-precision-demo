# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the `main` branch. It benchmarks two extended-precision backends side by side inside Kokkos CUDA kernels:

- **CUDA Emulated FP128** (`quad::cuda_fp128::fp128_t`): ~33 decimal digits, requires compute ≥ 10.0 (sm_100, Blackwell)
- **Kokkos DD (double-double)** (`quad::ddfun::ddouble`): ~30–31 decimal digits, portable to any CUDA GPU

Each operation runs on FP128, DD, and FP64. The output table shows **slowdown vs FP64** and **accuracy in decimal digits** for both backends side by side.

It produces two executables: `kokkos_ep_demo` (39 real math ops) and `kokkos_ep_demo_complex` (24 complex math ops).

## Branch structure

| Branch | Backend(s) | GPU requirement |
|---|---|---|
| `main` | CUDA FP128 + Kokkos DD | compute ≥ 10.0 (sm_100) |
| `CUDAFP128Kokkos` | CUDA FP128 only | compute ≥ 10.0 (sm_100) |
| `ddfunKokkos` | Kokkos DD only | any CUDA-capable GPU |

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
# All real operations — FP128 and DD side by side
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
    ├─ CUDA FP128 Backend
    │   (third_party/include/NVIDIA_emulated_quad/quad_math.hpp, quad_complex.hpp)
    ├─ Kokkos DD Backend
    │   (third_party/include/dd_math.hpp, dd_complex.hpp)
    └─ Kokkos Runtime + CUDA Backend
```

### CUDA FP128 Wrapper (`third_party/include/NVIDIA_emulated_quad/`)

`quad_math.hpp` defines `fp128_t` in `namespace quad::cuda_fp128`. Critical dual-path design:

```cpp
#ifdef __CUDA_ARCH__
    // Device: calls NVIDIA __nv_fp128_* functions (sm_100+ only)
#else
    // Host: uses native C++ operators / quadmath.h
#endif
```

`quad_complex.hpp` defines `quad_complex {fp128_t re; fp128_t im;}` with full math in `quad::cuda_fp128`.

### Double-Double Math Library (`third_party/include/`)

`dd_math.hpp` defines `ddouble { double hi; double lo; }` in `namespace quad::ddfun`. Key algorithms:

- **TwoSum (Knuth)**: basis of `ddadd` / `ddsub` — error-free summation
- **TwoProduct (Dekker splitting)**: basis of `ddmul` / `dddiv` — no FMA required
- **Constants**: constructed from IEEE 754 bit patterns via `make_dd()` for both host (`std::memcpy`) and device (`__longlong_as_double`)

All functions are `KOKKOS_INLINE_FUNCTION`, ported from DDFUN (David H. Bailey, Lawrence Berkeley National Lab).

`dd_complex.hpp` defines `ddcomplex { ddouble re; ddouble im; }` with full complex math via `quad::ddfun::` free functions.

### Demo Executables (`src/`)

Each demo follows the same pattern:
1. Parse args → generate operation-specific random inputs on host
2. Compute `quadmath.h` host reference (ground truth, ~34 digits)
3. Deep copy inputs to Kokkos device Views for FP128, DD, and FP64
4. Run 2 warmup + N timed `Kokkos::parallel_for` launches for all three backends
5. Deep copy FP128 and DD results back, compute per-element relative error → "digits of accuracy" (`-log10(rel_error)`, capped at 33.0 for FP128 / 31.0 for DD)
6. Compute slowdown = backend_time / fp64_time per statistic (min/max/median/mean)
7. Print formatted table: FP128 and DD slowdown + accuracy side by side

### Namespaces

- `quad::cuda_fp128` — CUDA emulated FP128 types and math
- `quad::ddfun` — Kokkos double-double types and math
- Within demos, aliased as `namespace fp128 = quad::cuda_fp128; namespace dd = quad::ddfun;`

### Platform Constraint

`libquadmath` (host reference) is x86_64 only. CMake enforces this — the project will not build on ARM or other platforms.
