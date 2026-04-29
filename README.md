# Kokkos extended-precision demo

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
