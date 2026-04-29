# Kokkos extended-precision demo

Many scientific and engineering applications — from numerical linear algebra to particle physics simulations — require arithmetic precision beyond what standard 64-bit IEEE double (FP64, ~16 decimal digits) can provide. Historically, teams needing higher precision on CPUs have reached for GCC's quadmath / libquadmath library, which provides software-emulated IEEE 128-bit quad precision (~33 digits) through the `__float128` type. On GPUs, however, quad precision has long been unavailable: CUDA device code had no equivalent, forcing developers to use workarounds or keep high-precision work on the host.

Two developments are changing this. First, NVIDIA introduced emulated FP128 device math functions in CUDA 12.8, requiring compute capability ≥ 10.0 (sm_100, Blackwell architecture), finally bringing `__float128`-class arithmetic into GPU kernels. Second, the alternative double-double (DD) approach — representing a value as an unevaluated sum of two FP64 numbers to achieve ~30–31 digits of precision — has a long history in portable high-precision software, most notably in David H. Bailey's DDFUN library, originally written in Fortran 90 as a quad-precision substitute for systems lacking hardware 128-bit support. The DD algorithms in DDFUN have now been ported to Kokkos, making them available across any CUDA-capable GPU regardless of compute capability.

This repository benchmarks both approaches side by side within Kokkos CUDA kernels: CUDA Emulated FP128 (where Blackwell hardware is available) and Kokkos DD (portable to any GPU), measuring performance overhead relative to FP64 and verifying accuracy against quadmath reference values. The longer-term goal is to contribute these extended-precision backends into the Kokkos library ecosystem, making portable high-precision GPU computing available to the broader HPC community.

---

Demonstrates two extended-precision backends running side by side inside Kokkos CUDA kernels:

| Backend | Type | Precision | GPU requirement |
|---|---|---|---|
| **CUDA Emulated FP128** | `quad::cuda_fp128::fp128_t` | ~33 decimal digits | compute ≥ 10.0 (sm_100, Blackwell) |
| **Kokkos DD (double-double)** | `quad::ddfun::ddouble` | ~30–31 decimal digits | any CUDA-capable GPU |

Each operation is run on both backends and on FP64. The output table shows **slowdown vs FP64** (min / max / median / mean across N timed repeats) and **accuracy in decimal digits** for FP128 and DD side by side.

## Branch structure

| Branch | Backend(s) | GPU requirement |
|---|---|---|
| `main` | CUDA Emulated FP128 + Kokkos DD | compute ≥ 10.0 (sm_100) |
| `CUDAFP128Kokkos` | CUDA Emulated FP128 only | compute ≥ 10.0 (sm_100) |
| `ddfunKokkos` | Kokkos DD (double-double) only | any CUDA-capable GPU |

## Executables

### `kokkos_ep_demo` — real ops
Runs all 39 real math operations on FP128, DD, and FP64. Reports:
- **Slowdown vs FP64**: ratio of backend time to FP64 time, min / max / median / mean
- **Accuracy**: decimal digits of precision vs `quadmath.h` reference (max 33.0 for FP128, 31.0 for DD)

### `kokkos_ep_demo_complex` — complex ops
Runs all 24 complex math operations on FP128, DD, and FP64. Each op prints two rows (real part, imag part); slowdown is shown only on the first row.

## Dependencies

- Kokkos ≥ 5.1 with CUDA backend
- CUDA 12.x+, GCC 13.x+
- **Compute capability ≥ 10.0** (sm_100) required for `main` and `CUDAFP128Kokkos` branches (CUDA Emulated FP128)
- `quadmath.h` / libquadmath (x86_64 only — host reference for accuracy measurement)

## Build

```bash
source scripts/prepare.sh          # load modules (Argonne systems)
source scripts/build_with_kokkos.sh <install-dir>
```

Or, if Kokkos is already installed:

```bash
cmake -B build -DCMAKE_PREFIX_PATH=<kokkos-install-dir>
cmake --build build -j$(nproc)
```

This outputs `build/kokkos_ep_demo` and `build/kokkos_ep_demo_complex`.

## Usage

```bash
# All real ops — FP128 and DD side by side vs FP64 baseline
./build/kokkos_ep_demo --batch 500000 --repeats 5

# Single real operation
./build/kokkos_ep_demo --op sin --batch 1000000 --repeats 5

# All complex ops
./build/kokkos_ep_demo_complex --batch 500000 --repeats 5

# Single complex operation
./build/kokkos_ep_demo_complex --op exp --batch 1000000 --repeats 5

# Convenience scripts
./scripts/run_all_ops.sh --batch 500000
./scripts/run_all_complex_ops.sh --batch 500000
```

Arguments: `--op <name>`, `--batch N` (default: 1,000,000), `--repeats N` (default: 5), `--seed N` (default: 12345).

## Sample output

```
batch=1000000  repeats=5  seed=12345  warmup=2  timing=kernel+fence

            |                      CUDA Emulated FP128                      |                   Kokkos DD (double-double)                   |
            |       Slowdown vs FP64        |       Accuracy (digits)       |       Slowdown vs FP64        |       Accuracy (digits)       |
------------+-------------------------------+-------------------------------+-------------------------------+-------------------------------+
            |  Min  |  Max  |  Med  | Mean  |  Min  |  Max  |  Med  | Mean  |  Min  |  Max  |  Med  | Mean  |  Min  |  Max  |  Med  | Mean  |
=----------=+-------------------------------+-------------------------------+-------------------------------+-------------------------------+
 add        |   1.5x|   1.4x|   1.5x|   1.5x|  33.00|  33.00|  33.00|  33.00|   1.0x|   1.0x|   1.0x|   1.0x|  31.00|  31.00|  31.00|  31.00|
------------+-------------------------------+-------------------------------+-------------------------------+-------------------------------+
 sub        |   1.4x|   1.4x|   1.4x|   1.4x|  33.00|  33.00|  33.00|  33.00|   1.0x|   1.0x|   1.0x|   1.0x|  31.00|  31.00|  31.00|  31.00|
------------+-------------------------------+-------------------------------+-------------------------------+-------------------------------+
```

## Supported operations

### Real (`kokkos_ep_demo`) — 39 ops, both backends
| Category | Operations |
|---|---|
| Arithmetic | `add sub mul div` |
| Unary math | `sqrt abs exp log exp2 exp10 expm1 log2 log10 log1p` |
| Trig | `sin cos tan asin acos atan` |
| Hyperbolic | `sinh cosh tanh acosh asinh atanh` |
| 2-input math | `pow hypot fmod remainder copysign fmax fmin fdim` |
| 3-input | `fma` |
| Rounding | `ceil floor round trunc` |

### Complex (`kokkos_ep_demo_complex`) — 24 ops, both backends
| Category | Operations |
|---|---|
| Arithmetic | `add sub mul div` |
| Unary | `abs conj sqrt exp log log10` |
| Trig | `sin cos tan asin acos atan` |
| Hyperbolic | `sinh cosh tanh asinh acosh atanh` |
| Power / construction | `pow polar` |
