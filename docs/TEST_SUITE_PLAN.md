# Kokkos extended-precision demo — exhaustive test suite plan

**Repo:** `ReetBarik/kokkos-extended-precision-demo`
**Date locked:** 2026-07-31 (session with Minion)
**Owner:** Reet (implementer: cluster-Claude, sequenced from this doc)

## Context and premises

### Repo state as of 2026-07-31

Five branches exist:

- `main` — DD (double-double) + CUDA emulated FP128. No tests. This is where
  everything eventually lands (DD + FF + QF, all benchmarked against
  quadmath).
- `ddfunKokkos` — DD only. No tests.
- `fffunKokkos` — **FF (float-float) already implemented.** Working demos
  for real and complex ops. Ad-hoc validation via `scripts/test_ffmul.cpp`
  (single-op standalone) and `scripts/probe_op.cpp` (debugging tool, not a
  gating test). `PORT_NOTES.md` (220 lines) documents five categories of
  precision bugs found during the DD→FF port and how they were fixed. Not
  merged to main yet.
- `qffunKokkos` — **does not exist yet.** Will hold QF (quad-float,
  4×FP32) as a mechanical port of QD's `qd_real.cc`.
- `CUDAFP128Kokkos` — CUDA emulated FP128 only. Sibling branch,
  **explicitly excluded** from the portable-test-suite plan because it's
  NVIDIA-only and non-portable outside Blackwell (sm_100). Independent
  benchmarking track.

### Precision targets

| Backend | Representation | Bits | Digits | Priority |
|---|---|---|---|---|
| DD  | 2×FP64 | ~106 | ~31.9 | ships on `main` |
| FF  | 2×FP32 | ~48  | ~14.4 | ships on `main` |
| QF  | 4×FP32 | ~96  | ~28.9 | ships on `main` |
| TF  | 3×FP32 | ~72  | ~21.7 | **SKIPPED** — hardest algorithmically, no distinctive niche between FF and QF |

### Oracle: Kokkos-wrapped quadmath (VERIFIED, adopted)

Kokkos ships `core/src/impl/Kokkos_QuadPrecisionMath.hpp` (~110 lines).
Provides `__float128` overloads in `namespace Kokkos` for every math
function needed as oracle: `abs, fma, exp, log, sqrt, pow, sin, cos, tan,
asin, acos, atan, atan2, sinh, cosh, tanh, asinh, acosh, atanh, erf,
tgamma, ceil, floor, round, trunc, ldexp, scalbn, copysign,
isinf, isnan, signbit`.

- Gated by `KOKKOS_ENABLE_LIBQUADMATH` CMake option (default ON on Linux
  per `cmake/kokkos_tpls.cmake`).
- Host-only: `inline`, not `KOKKOS_INLINE_FUNCTION`. Perfect for oracle
  role — libquadmath doesn't exist on GPUs, but oracle computation happens
  on host anyway.
- Linked automatically via `kokkos_link_tpl(kokkoscore PUBLIC LIBQUADMATH)`.
- Header is in `impl/` — private convention, may want to include via
  `Kokkos_MathematicalFunctions.hpp` if the overloads leak there when
  `KOKKOS_ENABLE_LIBQUADMATH` is set (verify at implementation time).

**Precision headroom check:**

- Oracle: 34 digits
- DD target: 31.9 digits → 2.1 digit headroom (tight but usable for
  accuracy; MAYBE insufficient for tight-bound verification)
- FF target: 14.4 digits → 20 digit headroom (comfortable)
- QF target: 28.9 digits → 5.1 digit headroom (adequate for pass/fail;
  MPFR ≥150 bits recommended as optional secondary oracle for tight-bound
  work)

MPFR is an optional secondary oracle behind a CMake flag; do not couple
default builds to it.

#### Complex oracle (T0.3)

Kokkos ships **no** `__complex128` wrapper upstream — `Kokkos_QuadPrecisionMath.hpp`
covers only real `__float128`. The complex demo (`src/demo_complex.cpp`) needs a
`__complex128` oracle (`cexpq`, `csqrtq`, `csinq`, …).

To keep the complex oracle plumbing symmetric with the real one (T0.0), this repo
carries a **local extension header**,
`impl/Kokkos_ComplexQuadPrecisionMath.hpp`, that adds `__complex128` overloads in
`namespace Kokkos` (`Kokkos::exp`, `Kokkos::sqrt`, `Kokkos::conj`, `Kokkos::real`,
…). It is applied to Reet's local Kokkos install (dropped into the Kokkos source
tree's `core/src/impl/` before build/install).

- A repo-side copy lives at `patches/kokkos_complex_quad_math.hpp` with a
  `patches/README.md` explaining what it is, that it is **not upstream**, how to
  apply it, and which Kokkos version it was tested against (5.1.0, LIBQUADMATH ON).
- Each overload is a **one-line forward** to the corresponding `::c<fn>q`
  function, so the wrapper is **bit-exact against `::c<fn>q` by construction**.
  Verified by `scripts/smoke_kokkos_complex_quad.cpp` (compares `Kokkos::exp`
  against `::cexpq` bit-for-bit).
- `CMakeLists.txt` probes the Kokkos install for this header
  (`check_cxx_source_compiles`) and warns-and-continues if absent — the same
  graceful-degradation posture used for LIBQUADMATH itself. Only
  `kokkos_ep_demo_complex` depends on it.
- Kept local (not sent upstream) until the test suite stabilizes (Reet's call).

### Test framework

Recommend CTest + a lightweight header (or GoogleTest if you don't mind
the dep). Do NOT continue the `printf + return code` pattern from
`test_ffmul.cpp` beyond that single file — unmaintainable at
6 layers × 3 backends = 18 test binaries. Framework decision is part of
T0.1.

## The six test layers

Every backend gets all six layers. Layers are independent and can be
implemented in parallel within a phase.

1. **EFT unit tests.** Test `twoSum`, Dekker `twoProd` (or `twoProdFMA`)
   at the primitive level. Ground truth: for FF the FP64 sum/product is
   exact and provable; for DD/QF the quadmath sum/product is exact.
   Assert bit-equality in higher precision. 10⁶ random + full corner
   corpus.

2. **Non-overlap invariant checks.** For every op output, assert
   `fl(hi + lo) == hi` bit-exactly (DD/FF), or the length-4 Priest
   invariant `|f_{i+1}| ≤ ½ ulp(f_i)` for QF. Instrument in each op's
   return path under a debug-build flag OR as a wrapper test that
   consumes op outputs. 10⁶ inputs per op.

3. **Property/identity tests.** `a - a == 0`, `a * 1 == a`,
   `sqrt(a)² ≈ a`, `exp(log(a)) ≈ a`, `sin² + cos² ≈ 1`, commutativity
   of `mul` where the algorithm is symmetric. No oracle needed. Fast.

4. **Differential accuracy vs. quadmath oracle.** Per op, measure
   `max(rel_err / u^N)` across 10⁶ random + corpus inputs. Compare
   against published bounds. For DDFUN the bounds don't cite cleanly
   from Joldes-Muller-Popescu (JMP uses `ieee_add`, DDFUN doesn't) — so
   for each op, comment must cite either a published bound or
   `"observed empirically, no proven bound"`. **No silent citation of
   inapplicable bounds.**

5. **FMA-contraction guard.** Compile EFT tests with `--fmad=false` on
   device and `-ffp-contract=off` on host; verify pass. Then with
   contraction on, verify no silent breakage. Precedent:
   `scripts/test_ffmul.cpp` uses `-ffloat-store` for the host case.

6. **End-to-end cancellation-prone kernels with known answers.**
   `√(x²+1) − x` for large x (compare to rearranged
   `1/(√(x²+1) + x)`), Σ 1/k² → π²/6, Machin's formula for π, partial
   sums of alternating series. Assert digit-count vs quadmath oracle.

## Corner-case corpus (used by every layer)

Host-side generator producing arrays of:

- Subnormals (esp. FP32 subnormals for FF/QF)
- ±0, ±inf, NaN (only where op semantics allow)
- Powers of two
- `nextafter` neighbors (u apart, 2u apart, 10u apart)
- Near-cancellation pairs (a, a·(1 + ε) for small ε)
- Huge/tiny magnitude mixes (crossing splitter overflow — see
  PORT_NOTES §4a: FF `exp` broke at input > 79 because
  `b * 8193.0f` overflowed when b ≈ 2¹¹⁵)
- Half-integer boundaries (see PORT_NOTES §4b: FF `ffnint` off-by-one
  because 2⁴⁷ magic-constant trick fails at 24-bit mantissa; affects
  `nint`, `floor`, `ceil`, `round`, `trunc`, `fmod`, sincos arg reduction)

**Explicit regression corpus from PORT_NOTES:**

- `exp` at input > 79 (FF splitter overflow, historical bug 4a)
- `remainder(68.379…, 3.5066…)` giving −1.7533 vs +1.7533 (FF `ffnint`
  off-by-one, historical bug 4b)
- `atanh(a)` for `|a| < 0.5` (cancellation branch, PORT_NOTES §3c)
- `sinh(a)`, `cosh(a)` for `|a| < 0.5` (Taylor branch, §3b)
- `sin(x)`, `cos(x)`, `tan(x)` near multiples of π (joint sin/cos
  doublings needed, §3a)

## PORT_NOTES §5 — conditioning limits (NOT bugs, DO NOT fail-gate on)

Test suite must classify these as "expected min drop", not treat them as
regressions:

- `sub`, `fdim`, `fma` under near-cancellation (unavoidable, matches FP64)
- `asin`, `acos` near `|a| = 1` (derivative → ∞)
- `atanh` near `|a| = 1` (similar)
- `remainder` near multiples of `b` (result → 0 with fixed absolute error)
- `exp` at output denormal range (FF `lo` falls into FP32 subnormal)
- `sin`/`cos` near `±π` (needs triple-float π for arg reduction — out of
  scope)

Suite should report **min AND mean digits** for each op. Fail gate on
mean; separately report min with expected-min annotations.

## Sequencing and task breakdown

Total: 23 tasks, ~3-4 focused weeks. Sequential within phases; parallel
within a phase after the first task lands.

### Phase 0 — Foundation (2 tasks + 1 pre-task)

**T0.-1: Extract FP128 to sibling branch; make `main` portable. (DONE)**

- Executed 2026-07-31. FP128 code confirmed intact on `CUDAFP128Kokkos`
  (headers byte-identical to `main`'s copies before removal).
- Removed `third_party/include/NVIDIA_emulated_quad/` from `main`.
- Stripped all FP128 device-backend paths from `src/demo_real.cpp` and
  `src/demo_complex.cpp` (namespace alias, Views, kernels, accuracy
  columns). Complex quadmath oracle calls (`cexpq`, `csqrtq`, …) left
  intact — they are the reference, not the FP128 backend.
- `main` now builds and runs DD-only on a Serial Kokkos (no CUDA
  required). New Serial BEFORE baseline captured for the T0.0 diff.
- Note: `scripts/build_with_kokkos.sh` still forces
  `Kokkos_ENABLE_CUDA` + `Kokkos_ARCH_BLACKWELL100`; that is the
  remaining CUDA coupling on `main` (not `CMakeLists.txt`), flagged as a
  follow-up.

**T0.0: Migrate quadmath oracle to Kokkos wrapper.**

- Remove `find_library(QUADMATH_LIBRARY ...)` and the x86_64 gate from
  `CMakeLists.txt`. Replace with detection of
  `Kokkos_ENABLE_LIBQUADMATH` and fail gracefully (skip tests, not
  configure error) if off.
- Include Kokkos's quadmath overloads
  (`impl/Kokkos_QuadPrecisionMath.hpp` or its public umbrella if
  exposed via `Kokkos_MathematicalFunctions.hpp`).
- Convert all `expq`/`logq`/`sinq`/... calls in `src/demo_real.cpp` and
  `src/demo_complex.cpp` to `Kokkos::exp((__float128)…)` style.
- Verify demo accuracy numbers are byte-identical after the refactor
  (regression check).
- Deliverable: `main` builds and runs identically, but oracle plumbing
  is now via Kokkos.
- Do on `main` first, then cherry-pick to `ddfunKokkos`, `fffunKokkos`.

**T0.1: Test harness skeleton (backend-parameterized).**

- Create `tests/` directory, wire into `CMakeLists.txt` with
  `enable_testing()` and `add_test`.
- Pick and adopt test framework (CTest + lightweight header, or
  GoogleTest — document choice).
- Design `test_utils.hpp` with:
  - Backend type traits template (Backend = DD | FF | QF), so a single
    test file can instantiate across backends.
  - Random input generators seeded reproducibly.
  - `rel_err<Backend>(dut_result, quad_reference)` and
    `digits_of_accuracy` computation.
  - Kokkos device-side test runner pattern that copies inputs to device,
    runs op in parallel_for, copies results back to host for oracle
    comparison.
  - Pass/fail reporting with min/mean/max digit stats.
- Deliverable: `tests/hello_test.cpp` exercises harness on a trivial
  identity for DD. Verification: `ctest` runs, passes.
- Port `scripts/test_ffmul.cpp` structure as a reference example
  (rename to `tests/reference_ffmul_pattern.cpp` — the pattern this
  harness generalizes).

**T0.4: Rename DD to `Kokkos::Experimental::DoubleDouble` for upstream-readiness. (DONE)**

- Executed 2026-08-01. Commit `49c87df`.
- Pure refactor, no arithmetic changes. Done BEFORE T0.2 lands corpus code so the
  eventual Kokkos PR is a mechanical namespace move, not a rewrite. See the new
  "Upstreaming considerations" section under "Deliverable at end of Phase 3".
- **Namespace + type.** `namespace quad::ddfun` → `namespace Kokkos::Experimental`;
  `struct ddouble` → `struct DoubleDouble`; `struct ddcomplex` →
  `struct DoubleDoubleComplex` (bespoke struct, NOT `Kokkos::complex<DoubleDouble>`
  yet — that integration is a separate future task).
- **STL-style free functions.** `ddadd→add`, `ddsub→subtract`, `ddmul→multiply`,
  `dddiv→divide` (all still reachable via operators); `ddmuld→multiply_scalar`,
  `dddivd→divide_scalar` (internal helpers); `ddneg→negate`;
  `ddnint→round_to_nearest_int`; `ddang→angle` (internal DDFUN atan2(y,x)
  primitive; public `atan2(y,x)` wrapper unchanged); `powi→pow_int`;
  `ddmuldd→two_prod`. Transcendentals already had STL names (exp/log/sin/…) and
  did not rename. Internal `ddexpint→expint`, `ddincgamma→incgamma`.
- **Constants.** `dd_pi()`→`DoubleDouble_pi()`, and likewise `DoubleDouble_e()`,
  `DoubleDouble_log2()`, `DoubleDouble_log10()`, `DoubleDouble_sqrt2()`,
  `DoubleDouble_euler_gamma()`. Chose the free-function form over a
  `constants::pi<DoubleDouble>()` template: mirrors Kokkos's existing M_PI-style
  accessors, reads shorter at the call site, and these are runtime-built from
  IEEE-754 bit patterns so they cannot be constexpr template variables.
- **Factory.** `make_dd(hi,lo)` → static `DoubleDouble::from_bits(hi,lo)`:
  namespaced to the type, discoverable, no free-function symbol.
- **ADL + `Kokkos::` re-exposure.** Every single/two-output math function is
  ADL-findable (argument namespace is `Kokkos::Experimental`) AND re-exposed under
  `namespace Kokkos` via one-line forwards at the bottom of each header, mirroring
  `impl/Kokkos_QuadPrecisionMath.hpp`'s `__float128` overloads — so
  `Kokkos::exp(dd)` works identically to `Kokkos::exp(double)` /
  `Kokkos::exp(__float128)`. `add`/`subtract`/`multiply`/`divide` are NOT
  re-exposed under `Kokkos` (operators + explicit ADL only).
- **SPDX headers.** Kokkos SPDX + copyright header added to
  `third_party/include/dd_math.hpp`, `dd_complex.hpp`; `patches/` copy already
  had one (T0.3). Downstream files (demos, tests, harness, scripts) deliberately
  left header-free.
- **Callers.** Demos and tests use `namespace dd = Kokkos::Experimental;` alias
  (ergonomic call-site sugar; short `dd::` names retained). `BackendTraits<DD>::type`
  = `Kokkos::Experimental::DoubleDouble`.
- **Regression gate.** BEFORE (`HEAD~1`) vs AFTER accuracy-columns-only diff of
  both demos (`--batch 500000 --repeats 5 --seed 12345`) is **empty** —
  byte-identical behavior (rule 7). `ctest` (hello_test) still passes.

**T0.2: Corner-case corpus. (DONE)**

- Executed 2026-08-01. Commit `5ffa52a`.
- **`tests/corpus.hpp` (new).** Pure DATA + a tiny iteration API, downstream-only
  (NO SPDX header — never upstreamed, unlike `dd_math.hpp`). Precision-parametric
  on the scalar type (`double` for DD, `float` for FF/QF later; only these two).
- **Return shape decision:** materialized `std::vector<T>` (unary) /
  `std::vector<std::pair<T,T>>` (binary), NOT the `InputDist` generator-functor
  shape. Rationale: corpus entries are deterministic constants (a specific
  subnormal, the literal 88.72 that broke FF `exp`), so a vector the caller
  iterates — one element == one deterministic test input — is the natural
  representation; generator functors model random draws, of which there are none
  here.
- **Categories:** subnormals (min/mid/largest denormal), ±0, ±inf, quiet NaN
  (opt-in), powers of two, `nextafter` neighbors (anchors 0/1/π/e/1e6/1e-6, ±2
  ulp), near-cancellation pairs, huge/tiny mixes (ratios 1e6..1e30),
  half-integer boundaries.
- **Two API styles (both shipped):** bundlers `unary<T>(flags)` /
  `binary<T>(flags)` (for T\*.2 invariant sweeps) and named accessors (for T\*.4
  accuracy tests so failures cite a specific PORT_NOTES bug). `CorpusFlags`
  (`include_nan=false`, `include_inf/zero/subnormals=true`) is authoritative over
  the *whole* assembled bundle (incidental members from other categories are
  filtered too).
- **PORT_NOTES §4/§3 regression accessors (verbatim named entries):**
  `exp_overflow` ({79.5,80,85,88.7,88.72}, §4a), `nint_half_integer`
  (19.4999993 + k±0.5 neighbors for k∈{0,1,2,10,100,1000,19}, §4b),
  `remainder_regression` ((68.379, 3.5066) + neighbors, §4b), `atanh_small`
  (§3c), `sinh_cosh_small` (§3b), `trig_near_pi` (±π/±2π/±3π/±π/2 neighbors, §3a).
  Category accessors (`subnormals`/`powers_of_two`/`nextafter_neighbors`/
  `finite_specials`/`zeros`/`infinities`/`nans`/`near_cancellation`/`huge_tiny`)
  are also individually callable.
- **`tests/test_utils.hpp` (edit).** `TODO(T0.2)` block replaced with an
  integration note; `#include "corpus.hpp"` added at file scope (corpus declares
  its own `namespace kokkos_ep::corpus`). Added corpus-pass runners
  `run_unary_op_on_corpus` / `run_binary_op_on_corpus` (same
  host→device→host→oracle pipeline as the random runners, driven by a corpus
  vector); the generator-based runners are unchanged for the random pass. Added
  the PORT_NOTES §5 expected-min-drop registry: `ExpectedMinDropAnnotation` +
  `lookup_expected_min_drop(op_name)`, preloaded with sub, fdim, fma, asin, acos,
  atanh, remainder, exp, sin, cos, tan. **Registry decision:** static constexpr
  table + linear scan — the set is tiny and fixed at compile time, so a table is
  allocation-free and reads as data next to `std::map` / an if/else chain.
- **`tests/corpus_test.cpp` (new).** Corpus-scaffolding smoke test (no DD op run,
  no oracle touched → no LIBQUADMATH guard needed). Verifies bundlers non-empty,
  flag behavior (inf/zero/subnormal present/absent per flag, NaN opt-in), the six
  named regression accessors non-empty with spot-checked values (88.72,
  19.4999993, (68.379,3.5066), |a|<0.5, near +π), and the registry
  (sub/fdim/fma/asin/acos/atanh/remainder non-null; add null).
  `kokkos_ep_add_test(corpus_test)` registered in `tests/CMakeLists.txt`.
- **Gate:** `cmake --build` clean; `ctest -V` shows hello_test AND corpus_test
  passing; `kokkos_ep_demo --batch 100` unchanged (no demo regression).

**T0.3: Add local Kokkos `__complex128` wrapper; route complex oracle
through Kokkos. (DONE)**

- Executed 2026-08-01. Commit `0b11585`.
- **A — header + install.** Wrote
  `impl/Kokkos_ComplexQuadPrecisionMath.hpp` (21 `__complex128` overloads in
  `namespace Kokkos`: `abs, real, imag, conj, exp, log, log10, pow, sqrt, sin,
  cos, tan, asin, acos, atan, sinh, cosh, tanh, asinh, acosh, atanh`), mirroring
  `Kokkos_QuadPrecisionMath.hpp` (SPDX header, include guards,
  `KOKKOS_ENABLE_LIBQUADMATH` gate, `#error` on missing `__float128`). Each
  overload is a one-line forward to `::c<fn>q`. `arg`/`cargq` omitted (demo does
  not use it). Repo-side copy saved to `patches/kokkos_complex_quad_math.hpp` +
  `patches/README.md`. Dropped into the local Kokkos 5.1.0 source tree and
  reinstalled; confirmed present at
  `<prefix>/include/impl/Kokkos_ComplexQuadPrecisionMath.hpp` (install's
  `FILES_MATCHING PATTERN "*.hpp"` picks it up automatically — no build-system
  change needed). Smoke test `scripts/smoke_kokkos_complex_quad.cpp` compiles,
  runs, and confirms `Kokkos::exp` is bit-exact against `::cexpq`.
- **B — demo migration.** `src/demo_complex.cpp` now includes
  `impl/Kokkos_ComplexQuadPrecisionMath.hpp` and routes all 21 oracle calls
  through `Kokkos::<fn>((__complex128)…)` (`cabsq→Kokkos::abs`,
  `conjq→Kokkos::conj`, `csqrtq→Kokkos::sqrt`, `cexpq→Kokkos::exp`,
  `clogq→Kokkos::log`, `clog10q→Kokkos::log10`, `csinq/ccosq/ctanq→sin/cos/tan`,
  `casinq/cacosq/catanq→asin/acos/atan`, `csinhq/ccoshq/ctanhq→sinh/cosh/tanh`,
  `casinhq/cacoshq/catanhq→asinh/acosh/atanh`, `cpowq→Kokkos::pow`,
  `crealq/cimagq→Kokkos::real/imag`). No DD-side arithmetic changed.
  `CMakeLists.txt` probes the install for the wrapper via
  `check_cxx_source_compiles` (linking `Kokkos::kokkos` so libquadmath resolves)
  and warns-and-continues if absent, pointing at `patches/README.md`. Only
  `kokkos_ep_demo_complex` is affected; `kokkos_ep_demo` (real) builds unchanged.
- **Regression gate.** BEFORE (`HEAD~1`) vs AFTER accuracy-columns-only diff of
  `kokkos_ep_demo_complex --batch 500000 --repeats 5 --seed 12345` is **empty**
  — behavior byte-identical, as required (rule 7).
- Bit-exactness is by construction (one-line forwards), so the refactor cannot
  change oracle values.

**T0.5: License hygiene — correct DDFUN attribution; add repo LICENSE /
NOTICE for dual-license posture. (DONE)**

- Executed 2026-08-01. Commit `e1b4de0`.
- **Problem fixed.** T0.4 had mechanically stamped Kokkos's
  `Apache-2.0 WITH LLVM-exception` SPDX onto `third_party/include/dd_math.hpp`
  and `dd_complex.hpp` to match Kokkos style. Those two files are C++/Kokkos
  ports of DDFUN v04 and are governed by David H. Bailey's **DHB-License**
  (modified BSD-3-Clause + §3 grant-back), not Kokkos's license. A Kokkos SPDX on
  a DDFUN derivative does not relicense DDFUN; it makes the file's status
  ambiguous. T0.5 corrects this and establishes the repo's overall dual-license
  posture.
- **Header rewrite.** Removed the two T0.4 SPDX lines from `dd_math.hpp` and
  `dd_complex.hpp`; replaced with a `LicenseRef-DHB-License` block (David H.
  Bailey copyright 2024 + "Modifications Copyright (c) 2026 UChicago Argonne,
  LLC"), DDFUN v04 provenance, and a modifications list. Additive only — no code
  inside the headers changed.
- **License files added.** Top-level `LICENSE` (Apache-2.0, verbatim from
  apache.org); `LICENSES/Apache-2.0.txt` (identical duplicate for SPDX tooling);
  `LICENSES/LicenseRef-DHB-License.txt` (verbatim byte-for-byte from
  davidhbailey.com, CRLF preserved, 24 lines,
  SHA-256 `8e167fe5…5fa5200`); `NOTICE.md` (plain-English per-file mapping, §3
  grant-back explanation, contacts).
- **`patches/kokkos_complex_quad_math.hpp`** verified correct
  (`Apache-2.0 WITH LLVM-exception`, Kokkos-style extension, not a DDFUN
  derivative) and left unchanged. `patches/README.md` gained a "License" section
  explaining why it matches Kokkos's license.
- **This doc.** Added the "Licensing" section above (dual-license posture,
  rationale, §3 grant-back, file-to-license mapping, Phase 2/3 FF/QF lineage
  checkboxes, Argonne review note); corrected the "Upstreaming considerations"
  SPDX bullet, which had assumed the DDFUN files were Apache-2.0.
- **Scope-out.** No code changed; no outreach to David H. Bailey or Argonne
  tech-transfer (Reet's follow-ups); no touching of FF/QF headers on sibling
  branches (their license lineage is a Phase 2/3 kickoff item).
- **Gate.** `grep -rn "SPDX-License-Identifier"` returns exactly the expected
  set (two DHB-License headers, one Apache-2.0-WITH-LLVM-exception patch, plus
  the license-text files themselves). Build unchanged; `ctest -V` shows
  hello_test AND corpus_test passing; demo smoke test unchanged.

### Phase 1 — DD validation (6 tasks)

**T1.1: EFT unit tests for DD. (DONE)**

- Executed 2026-08-01. Commit `30d8a40`.
- **`tests/dd_eft_test.cpp` (new, ~470 lines).** Layer-1 EFT unit test for the
  two error-free transforms every DD op is built on: the twoSum inside
  `DoubleDouble` `add` (dd_math.hpp:178-185) and the Dekker twoProduct inside
  `multiply` (dd_math.hpp:197-211, ≡ `two_prod` dd_math.hpp:270-278).
- **EFT primitive extraction — mirror-and-comment.** `add`/`multiply` embed the
  transforms in longer sequences that also fold in the input `.lo` components; for
  EFT testing we want the transform of two RAW doubles. So `two_sum` /
  `two_prod_dekker` are duplicated into the test file (bit-identical to the
  embedded transforms when `.lo == 0`), each with a comment citing the source
  lines. `dd_math.hpp` is NOT modified (rule 4). The duplication doubles as
  documentation of exactly what is under test.
- **Ground truth `__float128` — provable, not approximate.** Exact FP64 sum ≤54
  bits, exact FP64 product ≤106 bits, both ≤ binary128's 113-bit mantissa, so
  widening operands and computing sum/product in `__float128` is exact. Assert
  `(float128)hi + (float128)lo == (float128)a {+,*} (float128)b` bit-exactly.
  Oracle via Kokkos LIBQUADMATH; runtime-SKIP (77) if absent (hello_test pattern).
- **Coverage.** Test A (twoSum) and Test B (Dekker twoProd) each run 4 corpora:
  10⁶ uniform in [-1e100,1e100], 10⁶ uniform in [-1,1], 10⁵ `|a|≫|b|` pairs
  (`b=a·2^-k`, k∈[1,60]), and the full `corpus::unary<double>()` cross-product
  (i<j). Result: **A tested 2,113,526 / 0 failures; B tested 2,110,183 / 0
  failures.** Test C: 14/14 named hard cases (exact cancellation, both-subnormal,
  ±0, Bailey's `1.0 + 2^-53` → `lo` exactly 2^-53, π·π/e·e/√2·√2 spot-checks).
  Test D: device parity via `parallel_for` (Serial here → reduces to host, as the
  spec allows; catches device FP differences on CUDA/HIP/SYCL) — 200,000/200,000.
- **Out-of-domain inputs are SKIPPED, not failed.** twoSum skips only non-finite
  pairs and overflowing sums. Dekker twoProduct additionally skips subnormal
  operands, splitter-overflow magnitudes (`|x| ≥ 2^996`, checked BEFORE the
  zero-product shortcut because `x·0` still splits `x`), and products that overflow
  or gradually underflow (error term would fall subnormal). These are documented
  limits of Dekker's method (Dekker 1971; Muller et al. HFPA §4.4), not defects in
  `multiply`. The domain predicate tests the EXACT (float128) product, not the
  rounded FP64 product (which flushes to 0 under gradual underflow). No unguarded
  splitter-overflow bug was found in `ddmul` itself.
- **FMA-contraction posture.** `tests/CMakeLists.txt` gains
  `kokkos_ep_add_eft_test(<name>)` = `kokkos_ep_add_test` + per-target
  `-ffp-contract=off` (GNU/Clang, `COMPILE_LANGUAGE:CXX`), `-fp-model=precise`
  (Intel), `--fmad=false` (nvcc, `COMPILE_LANGUAGE:CUDA`, gated on
  `Kokkos_ENABLE_CUDA`). Per-target, not global, so demos/other layers keep normal
  flags. Verified `-ffp-contract=off` present in the verbose compile line. Reused
  by FF (T2.1) and QF (T3.1).
- **Gate.** `cmake --build` clean, zero warnings from `dd_eft_test.cpp`;
  `ctest -V` shows hello_test, corpus_test, dd_eft_test all Passed;
  `kokkos_ep_demo --batch 100 --repeats 1` unchanged (no regression).
- **Scope-out.** Only the twoSum-in-`add` and Dekker-twoProd-in-`multiply` EFTs
  (Layer 1). Higher-level ops → T1.2/T1.3/T1.4. No FF/QF EFTs (T2.1/T3.1). No MPFR
  (quadmath is provably exact for DD EFTs). No T1.5 contraction matrix. Demos
  untouched.

**T1.2: Non-overlap invariant checks for DD.**

- For every op in `dd_math.hpp` (add/sub/mul/div/sqrt/exp/log/sin/cos/
  tan/asin/acos/atan/sinh/cosh/tanh/asinh/acosh/atanh/pow/exp2/exp10/
  expm1/log2/log10/log1p/round_to_nearest_int/pow_int/hypot/fmod/
  remainder/fma), run
  10⁶ inputs + corpus and assert `fl(hi + lo) == hi` bit-exactly on
  outputs.
- Report which op fails, with input bit patterns for the failing case
  (mirroring `scripts/probe_op.cpp` output format).
- Independent of T1.1.

**T1.3: Property/identity tests for DD.**

- `a - a == 0` (exact), `a * dd_one() == a` (exact),
  `sqrt(a)² ≈ a` (within ~2u²), `exp(log(a)) ≈ a`,
  `sin²(a) + cos²(a) ≈ 1`, commutativity `multiply(a,b) == multiply(b,a)`.
- No oracle needed. Fast (< 1 second at 10⁶ inputs).
- Independent.

**T1.4: Differential accuracy for DD vs quadmath.**

- For each op, measure `max(rel_err / u²)` where `u = 2⁻⁵³`, across 10⁶
  random + corpus.
- Compare against DDFUN or QD published bounds where applicable. Where
  DDFUN-specific bounds are not in literature, cite observed max
  empirically with a comment noting no proven bound is available.
- Deliverable: table of `op | observed max u² | published bound | pass/fail`.
- Report min AND mean digits; annotate conditioning-limited ops
  (PORT_NOTES §5).
- Depends on T0.1, T0.2.

**T1.5: FMA-contraction guard for DD.**

- Compile T1.1 tests with `--fmad=false` (CUDA) and `-ffp-contract=off`
  (host); verify pass.
- Then with contraction on, verify `multiply`'s Dekker splitting sequence
  doesn't silently fold. Test-only; production build unchanged.
- One-time infra + regression test.
- Independent.

**T1.6: End-to-end cancellation kernels for DD.**

- `√(x²+1) − x` for x ∈ {1e6, 1e10, 1e15}: compare to
  `1/(√(x²+1) + x)`. Native FP64 loses digits; DD should retain ~31.
- Σ 1/k² for k=1..N → π²/6 (with `N=10⁶` giving ~6 digits of the tail;
  compare DD sum's digit-count against quadmath sum).
- Machin's formula for π (or Machin-like), evaluated in DD, digit
  count vs `DoubleDouble_pi()`.
- Partial sums of alternating series known to lose digits in FP64.
- Deliverable: ~150 lines, all pass at DD's expected ~31 digit
  precision.

### Phase 2 — FF validation (6 tasks)

FF library is already implemented on `fffunKokkos`. Phase 2 = validate
+ merge, not implement.

**T2.0: Merge fffunKokkos into main behind FF namespace.**

- Fast-forward or PR-merge `fffunKokkos` into `main`.
- Resolve any conflicts with T0.0 changes (both touched
  `CMakeLists.txt` and `demo_*.cpp`).
- Ensure `main` builds all three demos: `kokkos_ep_demo` (DD real),
  `kokkos_ep_demo_complex` (DD complex), `kokkos_ep_demo_ff`
  (FF real), `kokkos_ep_demo_ff_complex` (FF complex). Adjust names
  as needed.
- Update `README.md` and `CLAUDE.md` to reflect FF backend on main.
- Do NOT modify `ff_math.hpp` or `ff_complex.hpp` — the port has been
  validated empirically via PORT_NOTES bug-hunt; changes would need
  new testing.
- Deliverable: `main` HEAD builds and runs both DD and FF demos.

**T2.1: EFT unit tests for FF.**

- Test `ffadd` twoSum and `ffmul` Dekker (splitter `8193.0f` = 2¹²+1).
- Ground truth: **FP64** (provable — FP32 sum/product fits exactly in
  FP64's 53-bit mantissa; 24+24=48 ≤ 53). This is a *stronger* oracle
  than DD's quadmath because it's algebraically exact, not merely
  higher-precision.
- Port `scripts/test_ffmul.cpp` verbatim as the seed for this file,
  then generalize to `ffadd` and to the corner corpus.
- 10⁶ random + corpus.

**T2.2: Non-overlap invariant checks for FF.**

- Same as T1.2 but for `ff_math.hpp` ops (add/sub/mul/div/sqrt/exp/
  log/sin/cos/tan/... — inventory from `ff_math.hpp` at
  implementation time, ~40 ops).
- **Explicit regression tests from PORT_NOTES §4:**
  - `exp` with input > 79 must NOT return NaN
  - `ffnint(19.4999993...)` must return 19, not 20
  - `remainder(68.379..., 3.5066...)` must return +1.7533, not −1.7533
- Report which op fails with input bit patterns.

**T2.3: Property/identity tests for FF.**

- Same identities as T1.3, adjusted for FF's ~14 digit precision.
- No oracle needed.

**T2.4: Differential accuracy for FF vs quadmath.**

- Per-op `max(rel_err / u²)` where `u = 2⁻²⁴`.
- For FF, published bounds from CAMPARY / Joldes-Muller-Popescu apply
  more directly (FF is closer to canonical double-word than DD's
  DDFUN variant is). Cite where applicable.
- Report min AND mean; annotate PORT_NOTES §5 conditioning-limited
  ops as expected-min-drops.
- Expected mean: 13.3–14.0 digits per PORT_NOTES.

**T2.5: FMA-contraction guard for FF.**

- Same as T1.5 but for FF. Precedent: `test_ffmul.cpp` compiles with
  `-ffloat-store` for this reason.

**T2.6: End-to-end cancellation kernels for FF.**

- Same kernels as T1.6, expect ~14 digits accuracy.

### Phase 3 — QF implementation + validation (8 tasks)

QF does not exist yet. Phase 3 = build + validate. Model after QD's
`qd_real.cc`.

**T3.0a: QF library — arithmetic and renormalization.**

- Create branch `qffunKokkos` from `fffunKokkos` (inherits FF
  infrastructure).
- Follow the T0.4 naming convention: type + math under
  `namespace Kokkos::Experimental`, STL-style free-function names
  (`add`/`subtract`/`multiply`/`divide`/`negate`/`round_to_nearest_int`/
  `pow_int`), constants as `QuadFloat_pi()` etc., `QuadFloat::from_bits(...)`
  factory, and a bottom-of-header `namespace Kokkos` re-exposure block so
  `Kokkos::exp(qf)` works. FF gets the same treatment in T2.0.
- Create `third_party/include/qf_math.hpp` with:
  - `QuadFloat` struct: 4 × float components (f0, f1, f2, f3).
  - `renorm_4` (length-5 → length-4 Priest normalization; port from
    QD Hida-Li-Bailey Algorithm 3).
  - `add` (QD's `sloppy_add` + optional `ieee_add`; both from QD).
  - `subtract`.
  - `multiply` (16 partial products, keep down to weight u³, `renorm_4`).
  - `divide` (Newton, 3 iterations from FP32 → ~96 bits).
  - `sqrt` (Newton, 3 iterations).
  - `negate`, `abs`.
  - Constants (`QuadFloat_pi`, `QuadFloat_e`, `QuadFloat_log2`,
    `QuadFloat_log10`, `QuadFloat_sqrt2`, `QuadFloat_euler_gamma`) —
    regenerate from MPFR or quadmath as 4 × FP32 bit patterns (extend
    `scripts/gen_ff_constants.cpp` pattern).
- Every function `KOKKOS_INLINE_FUNCTION`, same style as
  `dd_math.hpp` and `ff_math.hpp`.
- **Preamble mandate: cluster-Claude must read `qd/src/qd_real.cc`
  from QD 2.3.24 upstream BEFORE porting.** Do not hallucinate QD
  internals. Cite QD source location in comments.
- Deliverable: `qf_math.hpp` compiles, arithmetic ops work, minimal
  standalone smoke test (like `scripts/test_ffmul.cpp` but for QF).

**T3.0b: QF library — transcendentals.**

- Add to `qf_math.hpp`:
  - `exp` (Taylor with argument reduction; more terms than QD's FP64
    version — derive term count for FP32-error-accumulation ~28-digit
    target, document in comments).
  - `log` (Newton on `exp`).
  - `log10`, `log2`, `log1p`, `exp2`, `exp10`, `expm1`.
  - `sin`, `cos`, `tan`, `sincos` (argument reduction, joint sin/cos
    doublings — apply PORT_NOTES §3a lesson from FF).
  - `asin`, `acos`, `atan`, `atan2`.
  - `sinh`, `cosh`, `tanh` (Taylor branch for `|a| < 0.5` per
    PORT_NOTES §3b).
  - `asinh`, `acosh`, `atanh` (Taylor branch for `|a| < 0.5` per
    PORT_NOTES §3c).
  - `pow`, `pow_int`, `hypot`, `fmod`, `remainder`, `copysign`, `fmax`,
    `fmin`, `fdim`, `fma`, `ceil`, `floor`, `round`, `trunc`,
    `round_to_nearest_int`.
- **Apply PORT_NOTES lessons proactively:**
  - `round_to_nearest_int`: don't use magic-constant trick with 2⁹⁵ —
    FP32 mantissa won't absorb it. Convert to FP64 or FP128 for
    rounding, like FF's
    `ffnint` does.
  - `exp` scaling: use direct scaling for final `ldexp`, not
    `multiply(s, ldexpf(1, nz))` — splitter would overflow.
- Deliverable: `qf_math.hpp` complete, `demo_qf_real.cpp` and
  `demo_qf_complex.cpp` (adapted from FF equivalents) run and produce
  ~28-29 digits accuracy against quadmath.

**T3.1: EFT unit tests for QF.**

- Test `twoSum`, Dekker `twoProd` at primitive level (same as FF —
  QF reuses FF's primitives internally).
- If any tests already cover this from T2.1, cross-reference; add QF
  wrapper tests as needed.
- Also test `renorm_4` at the primitive level: input a random
  length-5 unnormalized expansion, verify output satisfies Priest
  invariant `|f_{i+1}| ≤ ½ ulp(f_i)` AND equals input as real number
  (within QF truncation threshold).

**T3.2: Non-overlap invariant checks for QF.**

- Priest length-4 invariant: `|f_{i+1}| ≤ ½ ulp(f_i)` for i = 0,1,2.
- Every QF op, 10⁶ inputs + corpus.
- Report failures with 4-component bit patterns.

**T3.3: Property/identity tests for QF.**

- Same identities as T1.3/T2.3, adjusted for QF's ~29 digit precision.

**T3.4: Differential accuracy for QF vs quadmath (with MPFR fallback).**

- Per-op `max(rel_err / u⁴)` where `u = 2⁻²⁴`.
- Quadmath oracle: 5-digit headroom, adequate for pass/fail.
- MPFR ≥150 bits as optional secondary oracle behind a CMake flag,
  used only when tight-bound verification is needed. Do not hardcode.
- Published bounds from Hida-Li-Bailey (`ieee_add`: 2u⁴, `mul`: 16u⁴,
  `div`: ~50u⁴, `sqrt`: ~30u⁴). Cite QD source paper location per op.
- Report min AND mean; annotate conditioning-limited ops.

**T3.5: FMA-contraction guard for QF.**

- Same as T1.5/T2.5 for QF's Dekker sequences.

**T3.6: End-to-end cancellation kernels for QF.**

- Same kernels as T1.6/T2.6, expect ~29 digits accuracy.
- Adversarial kernels stronger than FF/DD can express: `sub(a, a·(1 +
  1e-25))` should still resolve; near-π `sin`/`cos` should now be
  distinguishable from noise (though not fully accurate — see
  PORT_NOTES §5).

## Rules for cluster-Claude implementation

Bake into launcher wrapper for every task:

1. **One task = one PR = one branch.** No task modifies files owned by
   another concurrent task. Phase 2/3 tasks target their respective
   branches (`fffunKokkos` merges → `main`; `qffunKokkos` builds → merges → `main`).
2. **Every task starts by reading:**
   - This file (`projects/kokkos_ep_test_suite.md`).
   - `tests/README.md` (created by T0.1).
   - The relevant `*_math.hpp` for the backend under test.
   - `PORT_NOTES.md` (for anything touching FF, and as reference for QF).
   - **NEVER the whole repo.** Context budget is precious.
3. **Verification gate is a passing `ctest`, not "it looks right".**
   Task done when new test file(s) pass in CI.
4. **If a task discovers a bug in existing code, it reports it and
   stops.** It does NOT fix. Bug fixes are separate tasks with clear
   scope. (Exception: T3.0a/T3.0b, which are building QF from scratch,
   can fix bugs in their own code as they go.)
5. **Numeric bounds cited in tests must have a source comment.** Format:
   ```
   // Bound: 4u² for double-word mul (Joldes-Muller-Popescu 2017, Thm 5.1)
   ```
   or:
   ```
   // Bound: observed empirically, no proven bound available for DDFUN variant
   ```
   Halts the "sounds plausible" failure mode.
6. **Never hallucinate library internals.** T3.0a/T3.0b in particular:
   read `qd_real.cc` from QD 2.3.24 (or clone if needed) BEFORE writing
   `qf_math.hpp`. Cite line numbers in comments.
7. **Preserve existing behavior across refactors.** T0.0 in particular:
   after refactor, demos must produce byte-identical accuracy numbers.
   Verify with `diff` on run output.
8. **Report format at task completion:**
   - What was implemented (files created/modified, LOC).
   - What tests pass (ctest output snippet).
   - What was NOT done and why (scope-out decisions).
   - Any bugs found in existing code (with reproduction, no fix).
   - Any deviations from this plan (must be justified).

## Parallelism graph

```
T0.0 ──┬─→ T0.1 ─┬─→ T1.1 ─┬─→ (T1.2, T1.3, T1.5, T1.6 all parallel)
       │        └→ T0.2 ─┤                                  ↓
       │                 └→ T1.4                    Phase 1 done
       │                                                    ↓
       └─→ (parallel with T0.1 if API locked)      T2.0 → T2.1 → (T2.2..T2.6 parallel)
                                                                              ↓
                                                                     Phase 2 done
                                                                              ↓
                                                              T3.0a → T3.0b → T3.1 → (T3.2..T3.6 parallel)
```

Phases sequential. Within phase: EFT layer (T*.1) first, then everything
else parallel.

## Sizing

- Phase 0: 3 tasks, ~1-2 days.
- Phase 1 (DD): 6 tasks, ~1 week.
- Phase 2 (FF): 6 tasks, ~1 week.
- Phase 3 (QF): 8 tasks, ~2 weeks (T3.0a/T3.0b are the biggest single
  tasks in the plan).

**Total: 23 tasks, ~3-4 focused weeks wall time.**

## Upstreaming considerations

The extended-precision types are being developed so that contributing them to
Kokkos later is a mechanical extraction rather than a rewrite. Locked in by T0.4
(DD) and inherited by FF (T2.0) and QF (T3.0):

- **Namespace + type names.** Types and their math functions live under
  `namespace Kokkos::Experimental` (`DoubleDouble`, and future `FloatFloat`,
  `QuadFloat`; complex as `DoubleDoubleComplex`, etc.). This is exactly where an
  upstream PR would place them, so the move is `git mv` + include-path fixups, not
  a symbol rewrite.
- **STL-style surface.** Arithmetic free functions use `add`/`subtract`/
  `multiply`/`divide`/`negate` (plus operators); transcendentals already match
  `<cmath>` (`exp`/`log`/`sin`/…); `round_to_nearest_int`, `pow_int`, `atan2`,
  and `two_prod` replace the DDFUN-Fortran `dd*` spellings. Constants are
  `DoubleDouble_pi()`-style free functions; bit-pattern construction is the
  static factory `DoubleDouble::from_bits(hi, lo)`.
- **ADL + `Kokkos::` re-exposure.** Every math function is ADL-findable via the
  argument's namespace AND re-exposed under `namespace Kokkos` (one-line
  forwards at the bottom of each header, mirroring
  `impl/Kokkos_QuadPrecisionMath.hpp`). So `Kokkos::exp(dd)` works identically to
  `Kokkos::exp(double)` / `Kokkos::exp(__float128)`. Arithmetic
  (`add`/`subtract`/`multiply`/`divide`) is reached via operators + explicit ADL
  only — deliberately NOT re-exposed as `Kokkos::add`.
- **SPDX + copyright headers** on the future-upstream files. Corrected in T0.5:
  the DDFUN-derived files (`third_party/include/dd_math.hpp`, `dd_complex.hpp`)
  carry the **DHB-License** (`SPDX-License-Identifier: LicenseRef-DHB-License`),
  NOT Kokkos's Apache-2.0 WITH LLVM-exception — they are ports of DDFUN and
  cannot be relicensed by us (see the "Licensing" section below). The
  Kokkos-style extension header `patches/kokkos_complex_quad_math.hpp` is the one
  file that genuinely carries `Apache-2.0 WITH LLVM-exception` (matching Kokkos
  for upstream compatibility). Whenever these types are upstreamed, the DDFUN
  attribution and DHB-License terms travel with the DDFUN-derived files; a Kokkos
  PR would need Kokkos maintainers' acceptance of the DHB-License for those two
  headers (or a clean-room reimplementation). Downstream-only files (demos,
  benchmarks, tests, this repo's harness, scripts) are deliberately header-free
  and covered by the top-level `LICENSE` (Apache-2.0).
- **Test structure.** Test logic is written against `BackendTraits<Backend>` so
  it could later drop into Kokkos's GoogleTest suite with minimal rewrite.
  Free-form `printf` is confined to the harness itself (`test_utils.hpp`), not
  the per-op test logic.
- **PR scope when it happens:** types + math functions + unit tests only. The
  demos, benchmarks, and this repo's CTest harness stay downstream. Opening a
  Kokkos PR is a future decision, made only after the test suite proves value.

## Licensing

Locked in by T0.5. This repository is **dual-licensed** (see `NOTICE.md` for the
authoritative, user-facing summary; this section is the rationale).

### Posture

- **Apache-2.0** — all Reet-authored original work: demos, tests, harness,
  corpus, scripts, docs. Covered by the top-level `LICENSE` (mirrored at
  `LICENSES/Apache-2.0.txt` for SPDX tooling). These files carry NO per-file SPDX
  header; the top-level `LICENSE` covers them.
- **DHB-License** (`LicenseRef-DHB-License`) — the two DDFUN-derived headers,
  `third_party/include/dd_math.hpp` and `dd_complex.hpp`. Full verbatim text in
  `LICENSES/LicenseRef-DHB-License.txt`.
- **Apache-2.0 WITH LLVM-exception** — `patches/kokkos_complex_quad_math.hpp`,
  the Kokkos-style complex-quadmath extension header. Matches Kokkos itself so it
  can be upstreamed verbatim.

### Why the split

DDFUN is authored by David H. Bailey and released under the **DHB-License**, a
modified BSD-3-Clause variant (Copyright (c) 2024 David H. Bailey). Our
`dd_math.hpp` / `dd_complex.hpp` are C++/Kokkos ports of DDFUN v04 — derivative
works. We cannot relicense DDFUN; the port inherits the DHB-License unchanged.
T0.4 had mechanically stamped Kokkos's Apache-2.0 WITH LLVM-exception onto these
two files to match Kokkos style; T0.5 corrected that, because a Kokkos SPDX on a
DDFUN derivative does not relicense DDFUN — it only makes the file's license
status ambiguous.

The complex-quadmath patch, by contrast, is a genuine Kokkos-style extension (a
companion to `Kokkos_QuadPrecisionMath.hpp`), not a DDFUN derivative, so it
correctly carries Kokkos's license.

### DHB-License §3 grant-back

The DHB-License adds a non-standard §3 to the usual BSD-3-Clause conditions: if
you publish modifications/enhancements to the DDFUN-derived files publicly
(which this repo does) without a separate written agreement, you thereby grant
David H. Bailey a non-exclusive, royalty-free, perpetual license to use, modify,
distribute, and sublicense those enhancements. In practice: anyone contributing
improvements to `dd_math.hpp` / `dd_complex.hpp` here grants Bailey the right to
fold them back into DDFUN, and anyone redistributing those files inherits this
term. (The DDFUN website also asks commercial users to contact the author at
`dhbailey@lbl.gov`; this is a courtesy pointer, not a term of the license text.)

### File-to-license mapping

| File(s) | License |
|---|---|
| `third_party/include/dd_math.hpp`, `dd_complex.hpp` | DHB-License (`LICENSES/LicenseRef-DHB-License.txt`) |
| `patches/kokkos_complex_quad_math.hpp` | Apache-2.0 WITH LLVM-exception |
| Everything else (demos, tests, harness, corpus, scripts, docs) | Apache-2.0 (`LICENSE`) |

### Phase 2/3 open question — FF and QF port lineage

FF (`fffunKokkos`) and QF (`qffunKokkos`) must each have their license verified
against their actual source tree **before** merging into `main`:

- If ported from **DDFUN** directly → inherits the **DHB-License** (Bailey,
  personal copyright, `dhbailey@lbl.gov`).
- If ported from **QD** (github.com/BL-highprecision/QD) → inherits the
  **different** `LBNL-BSD-License` (triple-authored Bailey/Li/Hida; LBNL
  *institutional* copyright; commercial contact `ipo@lbl.gov`, not Bailey
  personally).

These are distinct licenses with distinct copyright holders and contacts; the
correct header must be applied per source tree, not assumed uniform. Action
items:

- [ ] **T2.0 kickoff:** verify FF port lineage (DDFUN vs QD) and apply the
      correct license header before merging `fffunKokkos` into `main`.
- [ ] **T3.0a kickoff:** verify QF port lineage (DDFUN vs QD — QF is modeled on
      QD's `qd_real.cc`, so likely `LBNL-BSD-License`) and apply the correct
      license header before merging `qffunKokkos` into `main`.

### Argonne institutional review (Reet's follow-up)

Non-blocking for implementation, but must happen **before opening any Kokkos
PR**: a ~10-minute email to Argonne tech-transfer to confirm (a) Apache-2.0 for
the original code is approved, and (b) the DHB-License redistribution posture for
the DDFUN-derived files is acceptable. The "Modifications Copyright" holder used
for the ported files is "UChicago Argonne, LLC", the standard attribution form
for Argonne open-source releases.

## Deliverable at end of Phase 3

- `main` branch: DD, FF, QF all present, all benchmarked against Kokkos-
  wrapped quadmath oracle. `ctest` gates all three backends across all
  six test layers.
- Four backend branches maintained (`ddfunKokkos`, `fffunKokkos`,
  `qffunKokkos`, `CUDAFP128Kokkos`) — synced periodically with `main`
  for their respective backends; tests apply per-branch as relevant.
- `PORT_NOTES.md` extended with QF port notes (bugs found during T3.0
  development, per PORT_NOTES.md § template).
- README updated: three portable extended-precision backends
  (DD ~32 digits, QF ~29 digits, FF ~14 digits) benchmarked side-by-
  side against FP64 baseline, accuracy validated to published bounds.
- **Complex oracle dependency (T0.3).** The complex demos' `__complex128`
  oracle depends on the local Kokkos extension header
  `impl/Kokkos_ComplexQuadPrecisionMath.hpp`, which is **not upstream in
  Kokkos**. Future contributors must apply `patches/kokkos_complex_quad_math.hpp`
  per `patches/README.md` and rebuild Kokkos with
  `-DKokkos_ENABLE_LIBQUADMATH=ON` before building the complex demos; otherwise
  CMake warns and the complex-oracle compile fails.
