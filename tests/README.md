# `tests/` — extended-precision test suite

This directory holds the backend-parameterized test harness for the portable
extended-precision backends benchmarked in this repo. The authoritative spec is
[`docs/TEST_SUITE_PLAN.md`](../docs/TEST_SUITE_PLAN.md) — this README is the
operational quick-reference for the harness created in **T0.1**.

## Purpose

Each backend (DD today; FF, QF in later phases) is validated across six test
layers: EFT unit tests, non-overlap invariants, property/identity tests,
differential accuracy vs a `__float128` oracle, FMA-contraction guards, and
end-to-end cancellation kernels. The harness in `test_utils.hpp` provides the
shared plumbing so a single test file can target any backend without
duplication.

Two scaffolding tests exercise the harness itself: `hello_test.cpp`, a smoke test
that exercises the harness end-to-end (Kokkos init, input generation, host↔device
copy, oracle comparison, reporting) on a trivial DD round-trip identity; and
`corpus_test.cpp`, which validates the corner-case corpus (`corpus.hpp`) itself.
Neither runs a real DD math op.

Real DD correctness coverage begins in Phase 1 with `dd_eft_test.cpp` (**T1.1**),
the Layer-1 EFT unit test (see [EFT tests](#eft-tests-layer-1) below).

## Registered tests

| Test              | Layer / task | What it covers                                        |
|-------------------|--------------|-------------------------------------------------------|
| `hello_test`      | T0.1         | Harness plumbing on a trivial DD round-trip identity  |
| `corpus_test`     | T0.2         | Corner-case corpus (`corpus.hpp`) scaffolding         |
| `dd_eft_test`     | T1.1         | EFT bit-exactness: DD `twoSum` + Dekker `twoProduct`  |
| `dd_invariant_test` | T1.2       | Non-overlap invariant `fl(hi+lo)==hi` for **every** DD op (unary/binary/ternary/two-output); oracle-independent (no `__float128`, runs without LIBQUADMATH) |
| `dd_property_test` | T1.3        | Algebraic identities: **Group A** bit-exact (no oracle, e.g. `a·1==a`, `a-a==0`), **Group B** tolerance vs `__float128` (e.g. `sqrt(a)²≈a`, `sin²+cos²≈1`), **Test C** named-constant regressions (`sin(π)≈0`, …) |
| `dd_fma_guard_test` | T1.5       | FMA-contraction guard, **contraction OFF** — same Dekker `twoProduct` built `-ffp-contract=off`; **fail-gates** on any mismatch (stronger form of T1.1) |
| `dd_fma_guard_test_contract_on` | T1.5 | FMA-contraction guard, **contraction ON** — the *same source* built `-ffp-contract=fast`; **reports only** (always exits 0), prints the mismatch count and warns on drift vs `dd_fma_guard_baseline.txt` |

## How to run

Configure and build (from the repo root), then run the tests:

```bash
# Configure + build (Kokkos must be installed; see top-level README/CLAUDE.md)
cmake -B build -DCMAKE_PREFIX_PATH=<kokkos-install-dir>
cmake --build build -j

# Run the test suite
ctest --test-dir build -V
```

A passing run shows:

```
    Start 1: hello_test
1/1 Test #1: hello_test .......................   Passed
```

Tests build under `build/tests/`; you can also run a binary directly, e.g.
`./build/tests/hello_test`.

## How to add a new test

Five-line recipe:

1. Create `tests/foo_test.cpp`.
2. `#include "test_utils.hpp"` and write `main()` (init Kokkos, call the runners,
   use `KOKKOS_EP_ASSERT`, `return kokkos_ep::ep_exit_code();`).
3. Add one line to `tests/CMakeLists.txt`:  `kokkos_ep_add_test(foo_test)`.
4. Rebuild: `cmake --build build -j`.
5. Run: `ctest --test-dir build -V` — `foo_test` now appears.

`kokkos_ep_add_test(<name>)` compiles `tests/<name>.cpp`, links
`Kokkos::kokkos` + the third-party include dir, applies the quadmath define, and
registers the test with CTest (with `SKIP_RETURN_CODE=77`).

**Coverage recipe (Phase 1+):** a per-op test should run **two passes**. First a
random-generator pass via `run_unary_op` / `run_binary_op` (breadth). Then a
**corpus pass** via `run_unary_op_on_corpus` / `run_binary_op_on_corpus` fed from
`corpus.hpp` — prefer the **named accessor** for the op's known failure family
(so a failure cites a PORT_NOTES bug) and fall back to the `unary<T>()` /
`binary<T>()` bundler for broad invariant sweeps. When reporting the min digit
count, consult `lookup_expected_min_drop(op_name)` so PORT_NOTES §5
conditioning-limited ops report "expected-min-drop: OK" instead of failing.

## Backends and how tags map to types

The harness is templated over a **backend tag** via `BackendTraits<Tag>`:

| Tag  | Type            | Digits | Status                                  |
|------|-----------------|--------|-----------------------------------------|
| `DD` | `dd::DoubleDouble` | ~31 | supported (this branch)                 |
| `FF` | `ff::FloatFloat`   | ~14 | Phase 2 — adds `BackendTraits<FF>`      |
| `QF` | `qf::QuadFloat`    | ~29 | Phase 3 — adds `BackendTraits<QF>`      |

`DD` etc. are empty tag types, **not** the arithmetic types. `BackendTraits<DD>`
exposes `type` (`dd::DoubleDouble`), `u_squared`, `max_digits`, `name()`, and
`to_quad()`. A test written against `BackendTraits<Backend>` instantiates across
backends with no source duplication; Phase 2/3 add the FF/QF specializations.

## Corpus (`corpus.hpp`)

Uniform random inputs miss the pathological cases that actually break
extended-precision code — `PORT_NOTES.md` on branch `fffunKokkos` documents two
FF bugs (§4a `exp` NaN above input 79.4, §4b `ffnint` off-by-one at 19.4999…)
that slipped through the demo's own accuracy table precisely because random
almost never lands on them. `corpus.hpp` (T0.2) supplies a deterministic
corner-case corpus so, from Phase 1 onward, **every** test layer runs a random
pass *and* a corpus pass and those inputs are always exercised.

`corpus.hpp` is pure **data** plus a tiny API to iterate it — it is not tests. It
is downstream-only (no SPDX header, unlike `dd_math.hpp`). Entries are
precision-parametric: templated on the scalar type (`double` for DD today,
`float` for FF/QF later). Returns are materialized `std::vector<T>` (unary) /
`std::vector<std::pair<T,T>>` (binary), not `InputDist` generators — corpus
entries are fixed constants, so one vector element == one deterministic test
input.

**Categories:** subnormals, ±0, ±inf, quiet NaN (opt-in), powers of two,
`nextafter` neighbors, near-cancellation pairs, huge/tiny magnitude mixes,
half-integer boundaries. Plus the explicit PORT_NOTES §3/§4 regression families.

### Two API styles — pick per test

- **Bundlers** — `corpus::unary<T>(flags)` / `corpus::binary<T>(flags)`: "throw
  the whole corpus at this op." Use for the **T\*.2 invariant tests**, which just
  want broad coverage and don't care which category a failure came from. The
  `CorpusFlags` struct gates whole classes (`include_nan` defaults false;
  `include_inf`/`include_zero`/`include_subnormals` default true) and is
  authoritative over the *entire* assembled bundle, including members other
  categories emit incidentally (e.g. `nextafter(0,+inf)` = `denorm_min`).

- **Named accessors** — `corpus::exp_overflow<T>()`,
  `corpus::nint_half_integer<T>()`, `corpus::remainder_regression<T>()`,
  `corpus::atanh_small<T>()`, `corpus::sinh_cosh_small<T>()`,
  `corpus::trig_near_pi<T>()`, and the category accessors (`subnormals<T>()`,
  `powers_of_two<T>()`, …): grab exactly the family an op needs. Use for the
  **T\*.4 accuracy tests** so a failure cites a specific PORT_NOTES bug
  ("`exp_overflow` item 3") rather than "corpus item 47".

The corpus-pass **runners** live in `test_utils.hpp`:
`run_unary_op_on_corpus(inputs, host_oracle, device_op)` and
`run_binary_op_on_corpus(pairs, host_oracle, device_op)` — same
host→device→host→oracle pipeline as the random-pass runners, but driven by a
corpus vector instead of `(seed, n)` + generator.

### `lookup_expected_min_drop` — conditioning limits vs real regressions

Some ops legitimately show a low **min** digit count that is **not** a
regression: the operation is conditioning-limited and no fixed-precision
algorithm can do better (PORT_NOTES §5 — e.g. `sub`/`fdim`/`fma` under exact
cancellation, `asin`/`acos`/`atanh` near `|a|=1`, `remainder` near a multiple of
`b`, `exp` in the output-denormal range, `sin`/`cos`/`tan` near ±π). Tests
**fail-gate on the mean** column but must **not** fail on the min for these ops.

`test_utils.hpp` provides a registry:

```cpp
const ExpectedMinDropAnnotation* ann = lookup_expected_min_drop(op_name);
if (ann && stats.min >= ann->min_digits_allowed) {
  // "expected-min-drop: OK" — cite ann->reason
} else {
  // fail-gate on mean as usual
}
```

`lookup_expected_min_drop("sub")` returns non-null (with a `reason` string);
`lookup_expected_min_drop("add")` returns null (add is not conditioning-limited).

**Source of truth:** `PORT_NOTES.md` on branch `fffunKokkos` — §4 (two outright
bugs → the named regression accessors) and §5 (conditioning limits → the
expected-min-drop registry). The FF-side corpus specialization happens in Phase 2.

## EFT tests (Layer 1)

Layer 1 of the six-layer suite validates the two **error-free transforms** that
every double-word operation is built on:

- `twoSum(a, b) -> (s, e)`: `s = fl(a+b)` and `e = (a+b) - s` **exactly**.
- `twoProd_Dekker(a, b) -> (p, e)`: `p = fl(a*b)` and `e = a*b - p` **exactly**.

`dd_eft_test.cpp` (T1.1) tests these at the raw-`double` level — the twoSum
embedded in `DoubleDouble` `add` and the Dekker twoProduct embedded in `multiply`
(mirrored into the test file for RAW doubles; `dd_math.hpp` is not modified). If
either EFT is not bit-exact, nothing downstream (sqrt/exp/log/…) is trustworthy,
so this layer runs first. Ground truth is `__float128`, which is **provable, not
approximate**: the exact FP64 sum needs ≤54 bits and the exact FP64 product ≤106
bits, both of which fit in binary128's 113-bit mantissa, so widening the operands
and summing/multiplying in `__float128` is exact.

Inputs outside each transform's proven domain are **skipped, not failed**: twoSum
skips only non-finite pairs and sums that overflow; Dekker twoProduct additionally
skips subnormal operands, splitter-overflow magnitudes (`|x| ≥ 2^996`), and
products that overflow or gradually underflow (error term would fall subnormal) —
these are documented limits of Dekker's method, not defects in `multiply`.

### Contraction-off requirement

EFT tests **must** compile with FMA contraction disabled. Dekker's twoProduct
depends on `a1*b1 - c11` being two distinct rounded operations; if the compiler
fuses them into a single FMA, the error term collapses to zero and the transform
silently breaks — the test would then validate a transform the shipped binary does
not perform. `tests/CMakeLists.txt` provides a helper for this:

```cmake
kokkos_ep_add_eft_test(dd_eft_test)
```

`kokkos_ep_add_eft_test(<name>)` is `kokkos_ep_add_test(<name>)` plus per-target
contraction-off flags: `-ffp-contract=off` (GNU/Clang, on `COMPILE_LANGUAGE:CXX`),
`-fp-model=precise` (Intel), and `--fmad=false` (nvcc, on `COMPILE_LANGUAGE:CUDA`,
applied only when `Kokkos_ENABLE_CUDA`). Applied per target, not globally, so the
demos and other test layers keep the project's normal flags. Reuse this helper for
the future FF (T2.1) and QF (T3.1) EFT tests. (T1.5 later builds the full
contraction on/off regression matrix; T1.1 only needs the posture in place so its
own results are meaningful.)

## Non-overlap invariant (Layer 2)

Layer 2 (`dd_invariant_test`, **T1.2**) checks a **structural** property of every
DD op's *output* rather than its accuracy: a double-double `(hi, lo)` must be
**non-overlapping**, i.e. `lo` carries only bits below the last bit of `hi`. The
bit-exact statement of that, evaluated in **raw FP64** (a single hardware add +
compare), is

```
fl(hi + lo) == hi          (equivalently  |lo| <= 1/2 ulp(hi))
```

If `lo` held any bit at or above `hi`'s ulp, the rounded sum would land on a
different double and the equality would fail — localizing a normalization bug to
the exact op. Because this is a statement *about* FP64 rounding, the check is
deliberately **not** a `__float128` promotion (that would test the exact real
sum, a different thing). So this layer carries **no oracle and no
`KOKKOS_EP_HAVE_QUADMATH` guard**: it runs even on a quadmath-less Kokkos.
Accuracy-vs-oracle is the separate concern of T1.4.

**Coverage.** Every DD op that returns a double-double — unary, binary, ternary
(`fma`), two-output (`sincos`/`sinhcosh`, each output checked separately), and
`pow_int(dd,int)`. Each op runs two passes: 10^6 op-appropriate random inputs and
a full `corpus.hpp` pass (`include_zero=true`, `include_inf/nan=false`). Results
outside an op's domain (NaN/inf/subnormal `hi`, or out-of-domain input) are
**skipped, not failed**. Five ops (`add`, `multiply`, `sqrt`, `exp`, `sin`) also
run a device pass. Registered with the plain `kokkos_ep_add_test` helper — no
contraction flags (the invariant holds regardless of FMA contraction).

## Property/identity tests (Layer 3)

Layer 3 (`dd_property_test`, **T1.3**) checks **algebraic identities** the DD ops
must satisfy — a form of correctness orthogonal to Layer 1 (are the EFTs exact?)
and Layer 2 (are outputs well-formed?): *do the ops compose the way the algebra
says they should?* Identities are split by whether verifying them needs an oracle
at all.

**Group A — bit-exact, no oracle.** Identities whose two sides must produce the
*identical* `(hi, lo)` bit pattern, so the test is a raw `==` with no tolerance
and no `__float128`. This group runs unconditionally (even on a quadmath-less
Kokkos). Members: `add(a, negate(a)) == 0`, `a - a == 0`, `a·1 == a`,
`a·(-1) == negate(a)`, `abs` sign branches, `negate(negate(a)) == a`, and **add
commutativity** `add(a,b) == add(b,a)`. Add commutativity is bit-exact because
Knuth twoSum's error term is order-independent, so it stays in Group A.

**Group B — tolerance, needs the `__float128` oracle** (`#ifdef
KOKKOS_EP_HAVE_QUADMATH`; runtime-SKIP otherwise). Round-trip and trig
identities where both sides are correct but rounding makes them differ in the
last place(s): `sqrt(a)²≈a`, `exp(log(a))≈a`, `log(exp(a))≈a`,
`sin²+cos²≈1`, `sin(-a)==-sin(a)`, `cos(-a)==cos(a)`, `tan·cos≈sin`,
`2·sin·cos≈sin(2a)`, `exp(a)·exp(-a)≈1`, `hypot²≈a²+b²`, `pow(a,2)≈a·a`,
`atanh(a)≈½(log(1+a)−log(1−a))`, plus **multiply commutativity** — demoted here
from Group A because Dekker twoProduct's partial-sum ordering reorders under
operand swap (FP add is non-associative), so `multiply(a,b)` and `multiply(b,a)`
agree to ~31 digits but are **not** guaranteed bit-identical. Each identity is
scored with `digits_of_accuracy` and **fail-gates on the mean** against
`tolerance_digits = -log10(N·u²)` (`u = 2⁻⁵³`); per-identity proven bounds
(2u²/4u²/10u²) are cited in code comments (plan rule 5). Gating on the mean, not
the min, keeps conditioning-limited samples (e.g. `sin²+cos²` near `±π·k`, which
needs triple-float argument reduction) from false-failing.

**Test C — named-constant regressions.** Spot-checks that named constants /
transcendental round-trips hold to ≥30 digits: `|sin(π)|≈0` (softened via
`lookup_expected_min_drop("sin")`), `log(e)≈1`, `exp(log2)≈2`, `√2·√2≈2`,
`log(10)≈log10` constant. `euler_gamma`/digamma is **skipped** (no digamma op and
no independent DD oracle for the constant in `dd_math.hpp`).

**Device pass.** 3 Group A (`add_neg`, `mul_one`, `abs_branch`) + 2 Group B
(`sqrt_sq`, `pythag`) rerun on device (10⁵ inputs).

**Anti-tests (deliberately NOT tested, documented in the source).**
Associativity of `add` and distributivity across large-magnitude cancellations
are **false for any finite-precision format** (rounding is grouping-dependent) —
asserting them would be testing IEEE rounding, not the DD port. Registered with
the plain `kokkos_ep_add_test` helper (no contraction flags — not an EFT test).

## FMA-contraction guard (Layer 5)

Layer 5 (`dd_fma_guard_test`, **T1.5**) is the *positive* counterpart to the
defensive posture above. T1.1 builds `-ffp-contract=off` to protect its **own**
results; T1.5 asks whether that posture is actually **needed** by building the
identical Dekker `twoProduct` under **both** contraction settings and cross-
checking against a contraction-immune `__float128` oracle.

**One source, two targets.** `dd_fma_guard_test.cpp` is compiled twice:

```cmake
kokkos_ep_add_eft_test(dd_fma_guard_test)              # -> dd_fma_guard_test              (OFF, gates)
kokkos_ep_add_eft_test_contract_on(dd_fma_guard_test)  # -> dd_fma_guard_test_contract_on  (ON, reports)
```

Compiling the *same bytes* under different flags makes "identical inputs, identical
logic" a guarantee of the build system rather than a claim a reviewer must verify
across two drifting files. The only per-variant knobs are compile definitions the
helpers set: `KOKKOS_EP_CONTRACTION_MODE` (`0` = OFF/gate, `1` = ON/report) and,
for the ON variant, `KOKKOS_EP_BASELINE_PATH`.

`kokkos_ep_add_eft_test_contract_on(<name>)` mirrors `kokkos_ep_add_eft_test` but
forces contraction **on** into a distinct `<name>_contract_on` target:
`-ffp-contract=fast` (GNU/Clang), `-fp-model=fast` (Intel), and `--fmad=true`
(nvcc's default, stated explicitly). Both variants coexist because of the suffix.

**OFF variant — gates.** Asserts the Dekker error term is bit-exact (`F == 0`);
this is a stronger restatement of what T1.1 asserts, plus a `twoSum` control that
must stay exact (it has no contractible mul-then-± adjacency).

**ON variant — reports.** The compiler is *allowed* to contract; two outcomes,
both informative, neither a failure:

- `F == 0` — the compiler either did not contract, **or** contracted harmlessly.
  (On GCC 13.3.0 the latter holds: even with `-mfma` it emits 8 FMA instructions
  for the Dekker sequence, yet `F` stays 0 — Veltkamp splitting makes each partial
  product exactly representable, so fusing `partial ± accumulator` introduces no
  rounding difference. On this ISA target the `-ffp-contract=off` posture is
  belt+suspenders.)
- `F > 0` — the compiler contracted in a way that *does* change the result; the
  `-ffp-contract=off` posture in `dd_math.hpp`'s build is **required**, and `F` is
  the evidence.

The ON variant **always exits 0** — its value is the number, not a gate. A
*change* in `F` between builds is the regression signal, so the observed count is
recorded in `tests/dd_fma_guard_baseline.txt`; each ON run compares its live count
to that baseline and prints `baseline: OK` or `*** DRIFT ***` (a warning, never a
failure — investigate, then update the file if the new value is correct for the new
toolchain). **Scope:** the Dekker `twoProduct` only — the one DD primitive where
contraction is a documented hazard.

## Framework

CTest + the lightweight `test_utils.hpp` header — **no** GoogleTest/Catch2. The
sole assertion primitive is `KOKKOS_EP_ASSERT(cond, msg)`, which prints
`file:line` + message and makes `main()` return nonzero. Rationale (and the
option to revisit in Phase 1) is documented at the top of `test_utils.hpp`.

## Graceful degradation (no LIBQUADMATH)

The `__float128` oracle comes from Kokkos's quadmath overloads, available only
when Kokkos was built with `-DKokkos_ENABLE_LIBQUADMATH=ON`. When that TPL is
absent:

- CMake still configures and the tests still **build** (no configure error).
- `KOKKOS_EP_HAVE_QUADMATH` is left undefined, so the oracle code paths compile
  out.
- Oracle-dependent tests return exit code **77** at runtime, which CTest reports
  as **`Skipped`** (not failed) via `SKIP_RETURN_CODE`.

This is the same skip-don't-fail posture T0.0/T0.3 used for the demos: a
legitimately quadmath-less Kokkos config should not turn the suite red.
