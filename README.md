# Kokkos extended-precision demo

Demonstrates **CUDA FP128** extended-precision kernels built with **Kokkos**, comparing
device results against host **`quadmath.h`** references to verify accuracy.

## Executables

### `kokkos_ep_demo` — real ops
Benchmarks all available CUDA FP128 real math operations (`__nv_fp128_*`) and reports:
- **Timing**: kernel + `Kokkos::fence` (seconds), min / median / mean over N timed repeats
- **Accuracy**: number of precise decimal digits vs `quadmath.h` reference, min / max / mean / median over the batch (max = 33)

### `kokkos_ep_demo_complex` — complex ops
Benchmarks **`quad_complex`** (FP128) and **`Kokkos::complex<double>`** against the
`__complex128` quadmath reference. Reports timing and accuracy (precise digits) for the
**real** and **imaginary** parts in a single table, with two rows per operation.

## Dependencies

- Kokkos (≥ 5.1) with CUDA backend (Blackwell, sm_100)
- `quadmath.h` / libquadmath (host reference)
- CUDA FP128 (`__nv_fp128_*`, requires compute capability ≥ 10.0)

## Build

```bash
source scripts/build_with_kokkos.sh <install-dir>
```

## Usage

```bash
# Real ops demo
./build/kokkos_ep_demo --op sin --batch 1000000 --repeats 5

# Complex ops demo
./build/kokkos_ep_demo_complex --op exp --batch 1000000 --repeats 5

# Run all real ops
./scripts/run_all_ops.sh --batch 500000

# Run all complex ops
./scripts/run_all_complex_ops.sh --batch 500000
```

## Supported operations

### Real (`kokkos_ep_demo`)
| Category | Operations |
|---|---|
| Arithmetic | `add sub mul div` |
| Unary math | `sqrt abs exp log exp2 exp10 expm1 log2 log10 log1p` |
| Trig | `sin cos tan asin acos atan` |
| Hyperbolic | `sinh cosh tanh acosh asinh atanh` |
| 2-input math | `pow hypot fmod remainder copysign fmax fmin fdim` |
| 3-input | `fma` |
| Rounding | `ceil floor round trunc` |

### Complex (`kokkos_ep_demo_complex`)
| Category | Operations |
|---|---|
| Arithmetic | `add sub mul div` |
| Unary | `abs conj sqrt exp log log10` |
| Trig | `sin cos tan asin acos atan` |
| Hyperbolic | `sinh cosh tanh asinh acosh atanh` |
| Power / construction | `pow polar` |
