# Kokkos extended-precision demo

## Section 1 — Motivation

Many scientific and engineering applications — numerical linear algebra, particle
physics, long-running N-body and climate integrations, ill-conditioned solvers —
need arithmetic precision beyond what 64-bit IEEE double (FP64, ~16 decimal
digits) can provide. On host CPUs that need has historically been served by GCC's
libquadmath, which supplies software-emulated IEEE 128-bit quad precision through
the `__float128` type. That path does not travel: libquadmath is host-only and
x86_64-only, so code needing extra precision *inside* a portable compute kernel
has had nowhere to go.

This repository provides three portable, software-emulated extended-precision
backends that run inside Kokkos kernels — **DD** (double-double), **FF**
(float-float), and **QF** (quad-float). All three are written against Kokkos
alone, carry no hardware dependency, and compile for any Kokkos execution space:
CPU, GPU, and everything else Kokkos targets. Each is validated for accuracy
against a `__float128` (libquadmath) host oracle.

## Section 2 — Backends: types, ops, measured accuracy

### Precision types

| Backend | C++ type | Precision (approx. decimal digits) | Underlying math header |
|---|---|---|---|
| DD (double-double, 2×FP64) | `Kokkos::Experimental::DoubleDouble` | ~31 | `third_party/include/dd_math.hpp` |
| FF (float-float, 2×FP32) | `Kokkos::Experimental::FloatFloat` | ~14 | `third_party/include/ff_math.hpp` |
| QF (quad-float, 4×FP32) | `Kokkos::Experimental::QuadFloat` | ~29 | `third_party/include/qf_math.hpp` |

Every operation is documented in the header where it is defined; read the source
for algorithm choices, coefficient sources, and citations to the underlying DDFUN
/ QD references.

Complex layers are provided for all three backends — `DoubleDoubleComplex`,
`FloatFloatComplex`, `QuadFloatComplex` — mirroring the real-side type surface.
See the corresponding `*_complex.hpp` header for each.

### Operation inventory

All three real demos expose the same 39 real operations:

| Category | Operations |
|---|---|
| Arithmetic | `add sub mul div` |
| Unary math | `sqrt abs exp log exp2 exp10 expm1 log2 log10 log1p` |
| Trig | `sin cos tan asin acos atan` |
| Hyperbolic | `sinh cosh tanh acosh asinh atanh` |
| 2-input | `pow hypot fmod remainder copysign fmax fmin fdim` |
| 3-input | `fma` |
| Rounding | `ceil floor round trunc` |

All three complex demos expose the same 24 complex operations:

| Category | Operations |
|---|---|
| Arithmetic | `add sub mul div` |
| Unary | `abs conj sqrt exp log log10` |
| Trig | `sin cos tan asin acos atan` |
| Hyperbolic | `sinh cosh tanh asinh acosh atanh` |
| Power / construction | `pow polar` |

### Measured accuracy

Decimal digits of accuracy versus the `__float128` (libquadmath) host oracle,
per operation, one table per backend.

**Measurement conditions.** `--batch 1000000 --repeats 5 --seed 12345`, 2 warmup
launches, Kokkos 5.1 Serial execution space, GCC 13.3.0 / CMake 3.28.3, on an
AMD EPYC 7532 host. Figures are decimal digits, computed per element as
`-log10(|dev - ref| / |ref|)` and clamped to each backend's representable
ceiling — 31.00 for DD, 14.00 for FF, 29.00 for QF. A cell at the ceiling means
the result was correct to every digit the format can hold, not that error was
zero. Inputs are drawn per operation from an `mt19937_64` reseeded with `--seed`
each time, so a single-operation run reproduces the corresponding row exactly.

This table reports accuracy only; cost is treated separately under **Emulation
cost** below. Both sets of measurements come from a single-threaded Serial
build, which bounds what the timings can be read to mean — see the caveats
there.

These figures are optimization-invariant, and that was checked rather than
assumed: for each of the three backends the full 39-operation sweep was re-run
at `-O0` and at `-O3 -DNDEBUG`, and all 39 rows are identical across both
builds in every statistic. `CMAKE_CXX_EXTENSIONS OFF` means the backends
compile as strict ISO `-std=c++20`, where GCC leaves `-ffp-contract` off, so no
Dekker or TwoSum sequence gets contracted into an FMA. The `*_fma_guard_test`
targets pin that down directly — with contraction forced to `fast` at `-O3`,
they still observe zero incorrect error terms. Cost, by contrast, is *highly*
sensitive to optimization level; see the warning under **Emulation cost**.

**Statistic note.** The columns are the four statistics the demos compute
natively: min, max, median, mean. A p99 column was considered and is not
reported — the demos do not compute percentiles, and for an accuracy metric
(where higher is better) the 99th percentile reads the *best* tail, which sits at
the clamp ceiling for nearly every operation. `Min` is the worst observed element
across the batch and is the column to read for worst-case behaviour.

#### DD (double-double) — ceiling 31.00 digits

| Op | Min | Max | Median | Mean |
|---|---|---|---|---|
| `add` | 31.00 | 31.00 | 31.00 | 31.00 |
| `sub` | 31.00 | 31.00 | 31.00 | 31.00 |
| `mul` | 31.00 | 31.00 | 31.00 | 31.00 |
| `div` | 31.00 | 31.00 | 31.00 | 31.00 |
| `sqrt` | 31.00 | 31.00 | 31.00 | 31.00 |
| `abs` | 31.00 | 31.00 | 31.00 | 31.00 |
| `exp` | 29.29 | 31.00 | 30.42 | 30.44 |
| `log` | 30.86 | 31.00 | 31.00 | 31.00 |
| `exp2` | 29.33 | 31.00 | 30.40 | 30.43 |
| `exp10` | 29.28 | 31.00 | 30.36 | 30.40 |
| `expm1` | 29.16 | 31.00 | 31.00 | 30.68 |
| `log2` | 30.85 | 31.00 | 31.00 | 31.00 |
| `log10` | 30.81 | 31.00 | 31.00 | 31.00 |
| `log1p` | 30.90 | 31.00 | 31.00 | 31.00 |
| `sin` | 20.97 | 31.00 | 30.70 | 30.40 |
| `cos` | 24.47 | 31.00 | 30.54 | 30.45 |
| `tan` | 20.24 | 31.00 | 29.85 | 29.75 |
| `asin` | 19.47 | 31.00 | 29.93 | 29.75 |
| `acos` | 24.97 | 31.00 | 30.86 | 30.65 |
| `atan` | 29.93 | 31.00 | 30.88 | 30.78 |
| `sinh` | 26.35 | 31.00 | 30.41 | 30.43 |
| `cosh` | 29.33 | 31.00 | 30.45 | 30.47 |
| `tanh` | 29.35 | 31.00 | 31.00 | 30.93 |
| `acosh` | 30.76 | 31.00 | 31.00 | 31.00 |
| `asinh` | 30.54 | 31.00 | 31.00 | 31.00 |
| `atanh` | 25.35 | 31.00 | 30.47 | 30.38 |
| `pow` | 28.67 | 31.00 | 30.02 | 30.07 |
| `hypot` | 30.98 | 31.00 | 31.00 | 31.00 |
| `fmod` | 31.00 | 31.00 | 31.00 | 31.00 |
| `remainder` | 31.00 | 31.00 | 31.00 | 31.00 |
| `copysign` | 31.00 | 31.00 | 31.00 | 31.00 |
| `fmax` | 31.00 | 31.00 | 31.00 | 31.00 |
| `fmin` | 31.00 | 31.00 | 31.00 | 31.00 |
| `fdim` | 31.00 | 31.00 | 31.00 | 31.00 |
| `fma` | 31.00 | 31.00 | 31.00 | 31.00 |
| `ceil` | 31.00 | 31.00 | 31.00 | 31.00 |
| `floor` | 31.00 | 31.00 | 31.00 | 31.00 |
| `round` | 31.00 | 31.00 | 31.00 | 31.00 |
| `trunc` | 31.00 | 31.00 | 31.00 | 31.00 |

#### FF (float-float) — ceiling 14.00 digits

| Op | Min | Max | Median | Mean |
|---|---|---|---|---|
| `add` | 13.96 | 14.00 | 14.00 | 14.00 |
| `sub` | 8.81 | 14.00 | 14.00 | 13.96 |
| `mul` | 13.81 | 14.00 | 14.00 | 14.00 |
| `div` | 13.60 | 14.00 | 14.00 | 14.00 |
| `sqrt` | 13.51 | 14.00 | 14.00 | 14.00 |
| `abs` | 14.00 | 14.00 | 14.00 | 14.00 |
| `exp` | 10.42 | 14.00 | 13.43 | 13.41 |
| `log` | 13.92 | 14.00 | 14.00 | 14.00 |
| `exp2` | 12.13 | 14.00 | 13.39 | 13.41 |
| `exp10` | 12.18 | 14.00 | 13.37 | 13.40 |
| `expm1` | 12.35 | 14.00 | 14.00 | 13.79 |
| `log2` | 13.80 | 14.00 | 14.00 | 14.00 |
| `log10` | 13.78 | 14.00 | 14.00 | 14.00 |
| `log1p` | 13.92 | 14.00 | 14.00 | 14.00 |
| `sin` | 8.74 | 14.00 | 13.82 | 13.72 |
| `cos` | 8.16 | 14.00 | 13.80 | 13.72 |
| `tan` | 12.86 | 14.00 | 14.00 | 13.95 |
| `asin` | 12.21 | 14.00 | 13.97 | 13.84 |
| `acos` | 9.10 | 14.00 | 14.00 | 13.93 |
| `atan` | 13.70 | 14.00 | 14.00 | 14.00 |
| `sinh` | 12.33 | 14.00 | 13.66 | 13.63 |
| `cosh` | 12.53 | 14.00 | 13.68 | 13.65 |
| `tanh` | 12.56 | 14.00 | 14.00 | 13.96 |
| `acosh` | 13.99 | 14.00 | 14.00 | 14.00 |
| `asinh` | 13.72 | 14.00 | 14.00 | 14.00 |
| `atanh` | 12.64 | 14.00 | 14.00 | 13.92 |
| `pow` | 11.91 | 14.00 | 13.29 | 13.31 |
| `hypot` | 13.52 | 14.00 | 14.00 | 14.00 |
| `fmod` | 6.10 | 14.00 | 13.72 | 13.50 |
| `remainder` | 6.10 | 14.00 | 13.40 | 13.28 |
| `copysign` | 14.00 | 14.00 | 14.00 | 14.00 |
| `fmax` | 14.00 | 14.00 | 14.00 | 14.00 |
| `fmin` | 14.00 | 14.00 | 14.00 | 14.00 |
| `fdim` | 8.59 | 14.00 | 14.00 | 13.99 |
| `fma` | 9.78 | 14.00 | 14.00 | 13.99 |
| `ceil` | 14.00 | 14.00 | 14.00 | 14.00 |
| `floor` | 14.00 | 14.00 | 14.00 | 14.00 |
| `round` | 14.00 | 14.00 | 14.00 | 14.00 |
| `trunc` | 14.00 | 14.00 | 14.00 | 14.00 |

#### QF (quad-float) — ceiling 29.00 digits

| Op | Min | Max | Median | Mean |
|---|---|---|---|---|
| `add` | 29.00 | 29.00 | 29.00 | 29.00 |
| `sub` | 29.00 | 29.00 | 29.00 | 29.00 |
| `mul` | 28.27 | 29.00 | 29.00 | 29.00 |
| `div` | 27.65 | 29.00 | 29.00 | 28.99 |
| `sqrt` | 28.09 | 29.00 | 29.00 | 29.00 |
| `abs` | 29.00 | 29.00 | 29.00 | 29.00 |
| `exp` | 10.42 | 29.00 | 27.95 | 26.03 |
| `log` | 27.97 | 29.00 | 29.00 | 28.99 |
| `exp2` | 14.92 | 29.00 | 27.97 | 26.83 |
| `exp10` | 14.98 | 29.00 | 27.93 | 26.81 |
| `expm1` | 26.32 | 29.00 | 29.00 | 28.54 |
| `log2` | 27.96 | 29.00 | 29.00 | 28.99 |
| `log10` | 27.97 | 29.00 | 29.00 | 28.99 |
| `log1p` | 27.97 | 29.00 | 29.00 | 28.99 |
| `sin` | 23.29 | 29.00 | 28.62 | 28.56 |
| `cos` | 22.85 | 29.00 | 28.61 | 28.56 |
| `tan` | 27.41 | 29.00 | 29.00 | 28.88 |
| `asin` | 27.09 | 29.00 | 28.76 | 28.68 |
| `acos` | 24.22 | 29.00 | 29.00 | 28.85 |
| `atan` | 28.19 | 29.00 | 29.00 | 28.98 |
| `sinh` | 26.39 | 29.00 | 28.19 | 28.21 |
| `cosh` | 26.39 | 29.00 | 28.20 | 28.23 |
| `tanh` | 26.55 | 29.00 | 29.00 | 28.87 |
| `acosh` | 27.88 | 29.00 | 29.00 | 28.98 |
| `asinh` | 27.70 | 29.00 | 29.00 | 28.96 |
| `atanh` | 26.66 | 29.00 | 29.00 | 28.71 |
| `pow` | 25.79 | 29.00 | 27.71 | 27.76 |
| `hypot` | 28.12 | 29.00 | 29.00 | 29.00 |
| `fmod` | 29.00 | 29.00 | 29.00 | 29.00 |
| `remainder` | 29.00 | 29.00 | 29.00 | 29.00 |
| `copysign` | 29.00 | 29.00 | 29.00 | 29.00 |
| `fmax` | 29.00 | 29.00 | 29.00 | 29.00 |
| `fmin` | 29.00 | 29.00 | 29.00 | 29.00 |
| `fdim` | 29.00 | 29.00 | 29.00 | 29.00 |
| `fma` | 24.65 | 29.00 | 29.00 | 28.99 |
| `ceil` | 29.00 | 29.00 | 29.00 | 29.00 |
| `floor` | 29.00 | 29.00 | 29.00 | 29.00 |
| `round` | 29.00 | 29.00 | 29.00 | 29.00 |
| `trunc` | 29.00 | 29.00 | 29.00 | 29.00 |

The mean-gated regression versions of these measurements live in
`dd_accuracy_test`, `ff_accuracy_test`, and `qf_accuracy_test`, which assert a
per-operation floor rather than merely reporting numbers.

### Emulation cost

The useful question is not "how much slower than FP64 is extended precision" —
FP64 is not a substitute for a 29- or 31-digit type, so that ratio compares
things nobody chooses between. Each backend is instead measured against the
incumbent that delivers **comparable precision**:

| Backend | Digits | Incumbent it substitutes for | Why |
|---|---|---|---|
| FF | ~14 | FP64 (~16) | FF rebuilds FP64-class precision out of FP32, for hardware where FP64 is absent or rate-limited |
| DD | ~31 | `__float128` / libquadmath (~34) | libquadmath is the incumbent software path at this precision |
| QF | ~29 | `__float128` / libquadmath (~34) | same precision tier as DD |

Median of 5 repeats, batch 20000, seed 12345, Kokkos Serial, GCC 13.3.0,
`-O3 -DNDEBUG` (the default `CMAKE_BUILD_TYPE=Release`), AMD EPYC 7532. Inputs
are converted to each backend's type **before** the timed region, so what is
measured is the operation and not the cost of constructing the type. A ratio
below 1.00x means the backend is faster than the incumbent.

Each timed loop accumulates into a same-type running sum to keep the compiler
from discarding the work, so every row carries one extra add of its own type.
For the `add` and `sub` rows that is symmetric and harmless; elsewhere it
slightly flattens the ratios toward 1.00x, most noticeably on the cheap
arithmetic rows.

| Op | Class | FP64 ms | FF ×FP64 | `__float128` ms | DD ×f128 | QF ×f128 |
|---|---|---|---|---|---|---|
| `add` | arithmetic | 0.033 | 9.2× | 1.279 | 0.24× | 2.38× |
| `sub` | arithmetic | 0.033 | 10.3× | 1.394 | 0.25× | 2.33× |
| `mul` | arithmetic | 0.033 | 15.4× | 1.421 | 0.28× | 3.15× |
| `div` | arithmetic | 0.034 | 28.1× | 2.033 | 0.36× | 8.34× |
| `sqrt` | arithmetic | 0.095 | 10.4× | 19.951 | 0.05× | 3.24× |
| `fma` | arithmetic | 0.123 | 5.8× | 25.773 | 0.02× | 0.26× |
| `exp` | transcendental | 0.202 | 112.0× | 32.111 | 0.77× | 4.18× |
| `log` | transcendental | 0.116 | 235.5× | 19.333 | 2.28× | 15.04× |
| `sin` | transcendental | 0.418 | 50.0× | 18.110 | 0.79× | 8.56× |
| `cos` | transcendental | 0.447 | 47.0× | 17.952 | 0.80× | 8.66× |
| `atan` | transcendental | 0.215 | 320.4× | 11.446 | 4.30× | 48.03× |
| `pow` | transcendental | 0.342 | 117.1× | 48.505 | 1.19× | 8.00× |

Reproduce with `./build/kokkos_ep_bench_cost --batch 20000 --repeats 5`.

**Build the optimized configuration before reading anything into these
numbers.** The ratios are extremely sensitive to it. DD, FF and QF are
header-only `KOKKOS_INLINE_FUNCTION` code whose cost model assumes inlining;
libquadmath and libm are precompiled and optimized regardless of how this
project is built. An unoptimized build therefore penalises the backends and
leaves the incumbents untouched — measured at `-O0`, DD `add` reads 1.42×
instead of 0.24×, and the table inverts from "DD beats libquadmath" to "DD
loses to it". A stale build directory with `CMAKE_CXX_FLAGS_RELEASE` emptied
will reproduce exactly that error.

Reading the table:

- **DD beats libquadmath on arithmetic outright** — 0.24–0.36×, i.e. 3–4×
  faster — and by a wide margin on `sqrt` (20×) and `fma` (50×), where
  libquadmath's correctly-rounded software implementations are expensive. On
  transcendentals it splits: faster on `exp`, `sin`, `cos`, roughly par on
  `pow`, and slower on `log` (2.3×) and `atan` (4.3×). It gives up 3 digits
  against `__float128` and, unlike libquadmath, runs inside a device kernel.
- **Arithmetic and transcendentals are different regimes.** Kernels dominated
  by `+`, `*`, and `fma` — dot products, stencils, axpy, matrix products — live
  in the top half of the table, where DD is unambiguously the cheaper choice.
  The transcendental rows are not representative of that cost.
- **These are host numbers, and host is FF's and QF's worst case.** On this CPU
  an FP32 add costs the same as an FP64 add, so FF and QF pay for extra words
  with no compensating throughput. Their target is hardware where FP64 runs at
  1:32 or 1:64 against FP32; there, the FP64 baseline is itself penalised by
  that factor and the comparison inverts. On this EPYC, DD dominates QF on both
  speed and precision.
- **`__float128` is not a free reference point.** It implements full IEEE
  binary128 with correct rounding, subnormals, and exception semantics. DD and
  QF are unevaluated multi-word expansions with a narrower exponent range and
  no correct-rounding guarantee. A win against libquadmath is real but is not
  like-for-like.

#### Appendix: static op counts

Arithmetic volume per call, counted by walking the header implementation — not
a timing measurement. These are useful for comparing **backends to each other**
(QF `log` is 7.2× DD `log`), and for locating where an algorithm spends its
work. They are **not** a ratio against hardware FP64: by convention 3 below, a
libm call counts as 1 op, so an FP64 elementary function scores 1 against a
fully inlined multi-word expansion. Use the cost table above for that
comparison.

| Op | DD | FF | QF |
|---|---|---|---|
| `add` | 11 | 11 | 88 |
| `sub` | 12 | 12 | 92 |
| `mul` | 32 | 48 | 214 |
| `div` | 43 | 54 | 643 |
| `sqrt` | 47 | 61 | 3137 |
| `abs` | 3 | 3 | 5 |
| `exp` | 1213 | 1085 | 8814 |
| `log` | 3839 | 2326 | 27643 |
| `exp2` | 1245 | 1133 | 9028 |
| `exp10` | 1245 | 1133 | 9028 |
| `expm1` | 1226 | 1098 | 8907 |
| `log2` | 3882 | 2380 | 28286 |
| `log10` | 3882 | 2380 | 28286 |
| `log1p` | 3850 | 2337 | 27731 |
| `sin` | 1371 | 2086 | 15340 |
| `cos` | 1371 | 2086 | 15340 |
| `tan` | 1256 | 1934 | 14677 |
| `asin` | 4380 | 6893 | 52961 |
| `acos` | 4380 | 6893 | 52961 |
| `atan` | 4288 | 6771 | 49517 |
| `sinh` | 1335 | 1250 | 9647 |
| `cosh` | 1335 | 1250 | 9647 |
| `tanh` | 1309 | 1207 | 9643 |
| `acosh` | 3942 | 2459 | 31175 |
| `asinh` | 3941 | 2458 | 31171 |
| `atanh` | 3934 | 2448 | 28474 |
| `pow` | 5085 | 3460 | 36672 |
| `hypot` | 122 | 168 | 3653 |
| `fmod` | 129 | 138 | 952 |
| `remainder` | 113 | 122 | 976 |
| `copysign` | 6 | 6 | 10 |
| `fmax` | 2 | 2 | 2 |
| `fmin` | 2 | 2 | 2 |
| `fdim` | 14 | 14 | 94 |
| `fma` | 43 | 59 | 302 |
| `ceil` | 40 | 22 | 2 |
| `floor` | 41 | 23 | 2 |
| `round` | 26 | 8 | 27 |
| `trunc` | 42 | 24 | 3 |

**Counting convention.**

1. **What counts as one op.** Every `+`, `-`, `*`, `/`, `fma`, `sqrt`, and every
   comparison or select on a native-precision value counts as 1 op at the
   backend's baseline — FP64 for DD, FP32 for FF and QF.
2. **Error-free transforms are fully inlined.** A `two_sum` expands to its 6
   component ops, a `two_prod` to its 17. No structured breakdown appears in the
   table; each cell is a single integer. The repository compiles with FMA
   contraction off — the main `CMakeLists.txt` sets no contraction flag and GCC
   defaults to `-ffp-contract=off` for standard C++ — and the headers implement
   `two_prod` as an explicit Dekker split with no FMA path, so the non-FMA count
   applies throughout.
3. **libm calls count as 1 op** at baseline precision; this represents one
   hardware-approximate elementary function evaluation, not literal cycle cost.
   Where an operation seeds from libm and refines, the count is 1 + the
   correction ops.
4. **Constant-table loads are 0 ops** — they are loads, not arithmetic.
   Polynomial evaluation counts each individual multiply and add.
5. **Branches follow the general-case path** — the branch most finite,
   well-behaved inputs take for that operation's input range in the accuracy
   demo. Fast-path shortcuts for special values are not counted.
6. **Iterative refinement is counted at the iteration count the code actually
   runs.** Fixed loops count their trip count; convergence-checked loops count
   the typical case for the demo's input distribution. Series counts therefore
   depend on the reduced argument range and on each backend's convergence
   epsilon, which is why the same operation can differ across backends by more
   than the word count alone.
7. **Counts are positive integers**; no fractional counts are reported.

These are static algorithmic counts. They describe arithmetic volume, not
runtime — they ignore instruction-level parallelism, memory effects, and the
serial dependency chains that dominate renormalization-heavy code.

## Section 3 — Repository layout

```
third_party/include/
    dd_math.hpp        DD (2×FP64) real type + math — Kokkos C++ port of DDFUN v04
    dd_complex.hpp     DD complex layer (DoubleDoubleComplex)
    ff_math.hpp        FF (2×FP32) real type + math — mechanical DD→FF translation
    ff_complex.hpp     FF complex layer (FloatFloatComplex)
    qf_math.hpp        QF (4×FP32) real type + math — port of QD 2.3.24 quad-double
    qf_complex.hpp     QF complex layer (QuadFloatComplex)

patches/
    kokkos_complex_quad_math.hpp   Kokkos-style companion to Kokkos_QuadPrecisionMath.hpp

src/
    demo_real.cpp          DD real-operation demo      -> kokkos_ep_demo
    demo_complex.cpp       DD complex-operation demo   -> kokkos_ep_demo_complex
    demo_ff_real.cpp       FF real-operation demo      -> kokkos_ep_demo_ff
    demo_ff_complex.cpp    FF complex-operation demo   -> kokkos_ep_demo_ff_complex
    demo_qf_real.cpp       QF real-operation demo      -> kokkos_ep_demo_qf
    demo_qf_complex.cpp    QF complex-operation demo   -> kokkos_ep_demo_qf_complex
    bench_cost.cpp         tier-relative cost benchmark -> kokkos_ep_bench_cost

tests/                 23-test ctest suite covering all three backends
docs/                  TEST_SUITE_PLAN.md, PORT_NOTES_QF.md
scripts/               build helpers, coefficient generators, run-all scripts
PORT_NOTES.md          port-specific fixes and design lessons
LICENSE, NOTICE.md, LICENSES/   licensing — see Section 7
```

## Section 4 — Demos

Six executables. Each generates random inputs for the selected operation, runs
the extended-precision kernel and an FP64 kernel over the same data, and reports
timing alongside accuracy scored against the `__float128` host oracle.

- **`kokkos_ep_demo`** — DD, real operations. Reports slowdown versus FP64 and DD
  accuracy in decimal digits.
- **`kokkos_ep_demo_complex`** — DD, complex operations. Two rows per operation
  (real part, imaginary part).
- **`kokkos_ep_demo_ff`** — FF, real operations. Reports FF and FP64 kernel time
  in milliseconds side by side, plus accuracy for each.
- **`kokkos_ep_demo_ff_complex`** — FF, complex operations. Same two-rows-per-
  operation layout as the DD complex demo.
- **`kokkos_ep_demo_qf`** — QF, real operations. Same time-and-accuracy layout as
  the FF real demo, and additionally prints a per-operation pass verdict against
  a mean-accuracy gate.
- **`kokkos_ep_demo_qf_complex`** — QF, complex operations.

The DD executables are the un-suffixed ones because DD landed first; `_ff` and
`_qf` name the later backends.

Running them:

```bash
./build/kokkos_ep_demo_ff --batch 500000 --repeats 5
./build/kokkos_ep_demo_ff --op sin --batch 1000000 --repeats 5
```

Arguments: `--op <name>`, `--batch N` (default 1,000,000), `--repeats N`
(default 5), `--seed N` (default 12345). With no `--op`, every operation runs.

Convenience wrappers for whole-inventory sweeps:

```bash
./scripts/run_all_ops.sh              # DD real
./scripts/run_all_complex_ops.sh      # DD complex
./scripts/run_all_ff_ops.sh           # FF real
./scripts/run_all_ff_complex_ops.sh   # FF complex
./scripts/run_all_qf_ops.sh           # QF real
./scripts/run_all_qf_complex_ops.sh   # QF complex
```

### Cost benchmark

A seventh executable, **`kokkos_ep_bench_cost`**, is not a demo: it produces the
**Emulation cost** table in Section 2. It times each backend against the
incumbent at comparable precision — FF against FP64, DD and QF against
`__float128` — rather than against FP64 across the board, and converts inputs to
each backend's type before the timed region so the measurement is the operation
and not the conversion.

```bash
./build/kokkos_ep_bench_cost --batch 20000 --repeats 5
./build/kokkos_ep_bench_cost --op log
```

Arguments: `--op <name>`, `--batch N` (default 20,000), `--repeats N` (default
5), `--seed N` (default 12345). Twelve operations are covered — six arithmetic
(`add sub mul div sqrt fma`) and six transcendental (`exp log sin cos atan
pow`) — chosen to span both cost regimes rather than to mirror the demos' full
39-operation inventory.

## Section 5 — Tests

The suite is 23 ctest tests spanning all three backends:

- **Accuracy** — per-backend differential accuracy against the `__float128`
  oracle: `dd_accuracy_test`, `ff_accuracy_test`, `qf_accuracy_test`.
- **Property / identity** — algebraic identities that must hold for the type:
  `dd_property_test`, `ff_property_test`, `qf_property_test`.
- **Invariant** — non-overlap of the component words in the multi-word
  representation: `dd_invariant_test`, `ff_invariant_test`, `qf_nonoverlap_test`.
- **Error-free transforms** — the `two_sum` / `two_product` primitives the
  arithmetic is built on: `dd_eft_test`, `ff_eft_test`, `qf_eft_test`.
- **FMA-contraction guards** — two postures each, one compiled with contraction
  off and one with it on: `dd_fma_guard_test`, `ff_fma_guard_test`,
  `qf_fma_guard_test`, plus their `_contract_on` variants.
- **End-to-end kernels** — cancellation-heavy kernels exercising the types in
  realistic reductions: `dd_e2e_test`, `ff_cancellation_test`,
  `qf_cancellation_test`.
- **Foundational** — `hello_test`, `corpus_test`.

All 23 tests pass on `main`.

Tests are exercised on the Serial Kokkos execution space; the type headers are
`KOKKOS_INLINE_FUNCTION` throughout so they compile for device execution spaces
(CUDA, HIP, SYCL, OpenMP-target), but device-space CI is not yet in place.

Full test-suite design and gate rationale lives in `docs/TEST_SUITE_PLAN.md`;
port-specific fixes and design lessons live in `PORT_NOTES.md`.

Running the suite:

```bash
cd build/tests && ctest
echo "RC=$?"
```

## Section 6 — Usage

### Build

With Kokkos already installed:

```bash
cmake -B build -DCMAKE_PREFIX_PATH=<kokkos-install-dir>
cmake --build build -j$(nproc)
```

On the JLSE testbed maintained by CELS at Argonne National Lab, load modules
first and let the helper script fetch and install Kokkos:

```bash
source scripts/prepare.sh                      # JLSE modules
source scripts/build_with_kokkos.sh <install-dir>
```

`scripts/prepare.sh` is JLSE-specific — it hardcodes JLSE module names and
paths. Other Argonne resources (Polaris, Aurora, Improv) will need their own
module recipe; the rest of the build flow is portable.

Kokkos raises the C++ standard to C++20 through its exported interface, and
`libquadmath` (the host accuracy oracle) is x86_64-only — CMake enforces the
platform requirement.

### Templated kernel

The three backends share a type surface, so one templated kernel body serves all
of them with no per-backend specialization:

```cpp
#include <Kokkos_Core.hpp>
#include "dd_math.hpp"
#include "ff_math.hpp"
#include "qf_math.hpp"

namespace ex = Kokkos::Experimental;

// The backends do not expose a uniform `operator double`, so narrow at the
// boundary with a small overload set — one per backend representation.
inline double to_double(ex::DoubleDouble v) { return v.hi + v.lo; }
inline double to_double(ex::FloatFloat  v) { return (double)v.hi + (double)v.lo; }
inline double to_double(ex::QuadFloat   v) {
  return (double)v.f0 + (double)v.f1 + (double)v.f2 + (double)v.f3;
}

// Templated kernel — same body for every extended-precision backend.
template <class T>
void reduce_sin_squared(int n, double& out) {
  T sum(0.0);
  Kokkos::parallel_reduce(
    "reduce_sin_squared", n,
    KOKKOS_LAMBDA(int i, T& acc) {
      T x = T(static_cast<double>(i) * 1e-6);
      T s = Kokkos::sin(x);
      acc = acc + s * s;
    },
    sum);
  // Narrow only at the boundary; keep the value as T for full-precision output.
  out = to_double(sum);
}

int main() {
  Kokkos::initialize();
  {
    double dd_out, ff_out, qf_out;
    reduce_sin_squared<ex::DoubleDouble>(1000000, dd_out);
    reduce_sin_squared<ex::FloatFloat>  (1000000, ff_out);
    reduce_sin_squared<ex::QuadFloat>   (1000000, qf_out);
  }
  Kokkos::finalize();
  return 0;
}
```

Each backend provides `T(double)` construction, `operator+`, `operator*`, and a
`Kokkos::`-namespace `sin` overload, which is what lets the kernel body stay
identical across the three. Conversion back to `double` is the one place they
differ: none of the three defines `operator double`, so the example narrows
through explicit per-type accessors (`hi`/`lo` for DD and FF, `f0`–`f3` for QF).

## Section 7 — Licensing

This repository is dual-licensed. Repository-default is Apache-2.0 (see
`LICENSE`). The DDFUN-derived headers carry `LicenseRef-DHB-License`; the
QD-derived QF headers carry `LicenseRef-LBNL-BSD-License`;
`patches/kokkos_complex_quad_math.hpp` is `Apache-2.0 WITH LLVM-exception` to
match Kokkos. Full mapping, license texts, and the plain-English explanation of
the DHB-License §3 grant-back clause live in `NOTICE.md` and `LICENSES/`.

| File | License |
|---|---|
| `third_party/include/dd_math.hpp` | `LicenseRef-DHB-License` |
| `third_party/include/dd_complex.hpp` | `LicenseRef-DHB-License` |
| `third_party/include/ff_math.hpp` | `LicenseRef-DHB-License` |
| `third_party/include/ff_complex.hpp` | `LicenseRef-DHB-License` |
| `third_party/include/qf_math.hpp` | `LicenseRef-LBNL-BSD-License` |
| `third_party/include/qf_complex.hpp` | `LicenseRef-LBNL-BSD-License` |
| `patches/kokkos_complex_quad_math.hpp` | `Apache-2.0 WITH LLVM-exception` |
| Everything else | `Apache-2.0` |

## Section 8 — References

- **DDFUN v04** — David H. Bailey.
  <https://www.davidhbailey.com/dhbsoftware/ddfun-v04.tar.gz>
- **QD 2.3.24** — Yozo Hida, Xiaoye S. Li, David H. Bailey (LBNL).
  <https://www.davidhbailey.com/dhbsoftware/qd-2.3.24.tar.gz>
- **Kokkos** — <https://github.com/kokkos/kokkos>
- **`Kokkos_QuadPrecisionMath.hpp`** — the Kokkos header that
  `patches/kokkos_complex_quad_math.hpp` extends, at
  `kokkos/core/src/impl/Kokkos_QuadPrecisionMath.hpp` in upstream Kokkos.

Repository owner: Reet Barik. DDFUN questions: David H. Bailey
(<dhbailey@lbl.gov>).
