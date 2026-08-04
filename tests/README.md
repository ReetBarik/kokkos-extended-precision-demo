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
| `ff_eft_test`     | T2.1         | EFT bit-exactness: FF `twoSum` + Dekker `twoProduct` (splitter `8193.0f` = 2¹³+1); **FP64 oracle** — exact, no LIBQUADMATH needed, runs unconditionally |
| `dd_invariant_test` | T1.2       | Non-overlap invariant `fl(hi+lo)==hi` for **every** DD op (unary/binary/ternary/two-output); oracle-independent (no `__float128`, runs without LIBQUADMATH) |
| `ff_invariant_test` | T2.2       | Non-overlap invariant `fl(hi+lo)==hi` for **every** FF op, evaluated in **raw FP32**; FP32-narrower domain predicates derived from `ff_math.hpp`; oracle-independent (no `__float128`, runs without LIBQUADMATH) |
| `dd_property_test` | T1.3        | Algebraic identities: **Group A** bit-exact (no oracle, e.g. `a·1==a`, `a-a==0`), **Group B** tolerance vs `__float128` (e.g. `sqrt(a)²≈a`, `sin²+cos²≈1`), **Test C** named-constant regressions (`sin(π)≈0`, …) |
| `ff_property_test` | T2.3        | FF analogue of `dd_property_test`: **Group A** bit-exact (7 identities, no oracle), **Group B** tolerance vs `__float128` (13 identities, **mean-gated** at −log10(N·u²) ≈ 8.45 for u=2⁻²⁴), **Test C** named-constant regressions (target ≥12 of FF's 14 digits); exp-guard-narrowed domains for the exp round-trips; runtime-SKIPs without LIBQUADMATH |
| `dd_accuracy_test` | T1.4        | Differential accuracy vs `__float128`: per-op digits of accuracy over 10⁶ random + corpus; **fail-gates on MEAN** ≥ −log10(N·u²) ≈ 25.91; PORT_NOTES §5 conditioning-limited ops report **EXPECTED-MIN-DROP** (gated on mean, not min); runtime-SKIPs without LIBQUADMATH |
| `ff_accuracy_test` | T2.4        | FF analogue of `dd_accuracy_test`: per-op digits of accuracy over 10⁶ random + corpus vs `__float128`; **fail-gates on MEAN** ≥ −log10(N·u²) ≈ 8.45 (u=2⁻²⁴, cap 14); FP32-narrower op domains from T2.2; shared PORT_NOTES §5 registry → **EXPECTED-MIN-DROP**; runtime-SKIPs without LIBQUADMATH |
| `dd_e2e_test`     | T1.6         | End-to-end cancellation kernels: √(x²+1)−x, Σ1/k², Machin's π, alternating harmonic — all quadmath-oracle-gated |
| `ff_cancellation_test` | T2.5    | FF analogue of `dd_e2e_test`: same four cancellation kernels (√(x²+1)−x, Σ1/k², Machin's π, alternating harmonic) scored in digits vs `__float128`/closed-form oracles; **mean-gated at 14−3 = 11.0** (FF's cap minus headroom); K1 naive-vs-stable compares FF against **FP32** (FF's base scalar) at x∈{1e2,1e4,1e6}; runtime-SKIPs without LIBQUADMATH |
| `dd_fma_guard_test` | T1.5       | FMA-contraction guard, **contraction OFF** — same Dekker `twoProduct` built `-ffp-contract=off`; **fail-gates** on any mismatch (stronger form of T1.1) |
| `dd_fma_guard_test_contract_on` | T1.5 | FMA-contraction guard, **contraction ON** — the *same source* built `-ffp-contract=fast`; **reports only** (always exits 0), prints the mismatch count and warns on drift vs `dd_fma_guard_baseline.txt` |
| `ff_fma_guard_test` | T2.5 | FF analogue of `dd_fma_guard_test`, **contraction OFF** — same Dekker `twoProduct` (splitter `8193.0f` = 2¹³+1) built `-ffp-contract=off`; **fail-gates** on any mismatch. **FP64 oracle** (exact — 48-bit product fits FP64's 53-bit mantissa), so runs **unconditionally** (no LIBQUADMATH gate, no SKIP-77) |
| `ff_fma_guard_test_contract_on` | T2.5 | FF analogue, **contraction ON** — the *same source* built `-ffp-contract=fast`; **reports only** (always exits 0), prints the mismatch count and warns on drift vs `ff_fma_guard_baseline.txt` |
| `qf_eft_test`     | T3.1         | EFT bit-exactness for QF: `qf_two_sum` / `qf_quick_two_sum` / `qf_two_prod` / `qf_two_sqr` (splitter `8193.0f` = 2¹³+1; **FP64 oracle**, exact, no LIBQUADMATH) **plus** the QF-unique `renorm_4` (len 5→4) and `renorm` (len 4→4): Priest non-overlap invariant + **exact FP64 value-preservation** on bounded-spread inputs, and a wide-spread `__float128` truncation check (rel ≤ 2⁻⁸⁸) behind the quadmath guard. Calls the **shipped** `qf_math.hpp` free functions directly (no mirror-and-comment). Contraction OFF |

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

### FF EFT (Layer 1, Phase 2)

`ff_eft_test.cpp` (T2.1) is the FF analogue of `dd_eft_test.cpp`. It tests the
same two transforms at the raw-`float` level — the twoSum embedded in `FloatFloat`
`add` (`ff_math.hpp:174-181`) and the Dekker twoProduct embedded in `multiply`
(`ff_math.hpp:193-207`, ≡ `two_prod` `ff_math.hpp:266-274`), mirrored into the test
file for RAW floats; `ff_math.hpp` is not modified. It reuses the same four corpora
(broad random `[-1e30f,1e30f]`, narrow random `[-1,1]`, `|a|≫|b|` with `k∈[1,20]`,
full `corpus::unary<float>()` cross-product), named hard cases, and device-parity
pass.

The one material difference from T1.1 is the **oracle**. For FF the ground truth is
plain **FP64**, not `__float128`: the exact FP32 sum needs ≤25 bits and the exact
FP32 product ≤48 bits, both of which fit in FP64's 53-bit mantissa, so widening the
operands and summing/multiplying in `double` is **algebraically exact** — a
*stronger* oracle than DD's quadmath (exact, not merely higher-precision), and one
that needs no external library. So `ff_eft_test` carries **no `KOKKOS_EP_HAVE_QUADMATH`
gate and no runtime SKIP-77** — it runs on every build.

FP32-specific domain skips (out-of-domain, **not** failures): twoSum skips only
non-finite pairs and sums that overflow FP32; Dekker twoProduct additionally skips
subnormal operands, splitter-overflow magnitudes (`|x| ≥ FLT_MAX / (2¹³+1) ≈
2^115`, derived from the `a * 8193.0f` split — this is PORT_NOTES §4a's `exp`
splitter-overflow mechanism), and products that overflow or gradually underflow
(error term `< 2⁻¹⁰²` would fall into FP32 subnormals). Registered with the same
`kokkos_ep_add_eft_test(ff_eft_test)` (contraction OFF); the contraction-ON reporter
mirror is T2.5.

> **Splitter naming.** The shipped FF splitter is `8193.0f`, which is **2¹³ + 1**
> (not 2¹² + 1 = 4097). The `ff_math.hpp:192` comment states this correctly; a stale
> "2^12+1" typo in the `ff_math.hpp` license header and in the T2.1 task text is
> noted but not fixed here (T2.1 does not modify `ff_math.hpp`).

### QF EFT (Layer 1, Phase 3)

`qf_eft_test.cpp` (T3.1) is the QF analogue of `ff_eft_test.cpp`. It tests the FP32
error-free transforms QuadFloat composes on, plus the QF-unique renormalization
primitives. **Two structural differences from T2.1:**

1. **No mirror-and-comment.** `ff_eft_test` had to *duplicate* FF's twoSum / Dekker
   twoProduct into the test because `ff_math.hpp` embeds them inside longer
   `add`/`multiply` sequences. `qf_math.hpp` instead **exposes** the shipped
   primitives as free functions in `Kokkos::Experimental` — `qf_two_sum`,
   `qf_quick_two_sum`, `qf_two_prod`, `qf_two_sqr` (`qf_math.hpp:118-158`) and
   `renorm` / `renorm_4` (`qf_math.hpp:182-257`) — so this test calls the **actual
   shipped code**, a strictly stronger check (a mirror can drift; a direct call
   cannot). `qf_math.hpp` is not modified (rule 4).
2. **`renorm_4` has no FF analogue.** FF's two-word type never renormalizes a wide
   expansion; QF's `renorm_4` (len 5→4) and `renorm` (len 4→4) are genuinely
   QF-unique surface, and their oracle strategy is T3.1-original.

**twoSum/twoProd oracle** is the same provable **FP64** as T2.1 (25-bit sum / 48-bit
product both fit FP64's 53-bit mantissa), so that half runs unconditionally with no
LIBQUADMATH gate. FP32 domain skips (splitter overflow `|x| ≥ FLT_MAX/8193 ≈ 2^115`,
subnormal operands, underflow tail `< 2⁻¹⁰²`) are inherited verbatim from
`ff_eft_test`. `qf_quick_two_sum` is tested with operands ordered `|a| ≥ |b|` (its
precondition); `qf_two_sqr` reuses the twoProd domain on `(a,a)`.

**`renorm` oracle (T3.1-original).** Two value-preservation regimes:

- **Bounded spread (exact FP64, unconditional):** input words drawn inside a common
  ≤29-bit exponent window, so the exact real sum spans ≤53 bits — it fits *exactly*
  in FP64 **and** within QF's 96-bit capacity, so `renorm` drops nothing and
  `(double)(Σ out) == (double)(Σ in)` is a **provable bit-equality**. This is the
  primary gate.
- **Wide spread (quadmath, behind the guard):** words span the full ~96-bit range so
  `renorm` genuinely truncates; the residual is checked against QF's truncation
  threshold **rel ≤ 2⁻⁸⁸** (256× margin above `u = 2⁻⁹⁶`, `qf_math.hpp:11`). SKIPs
  cleanly without LIBQUADMATH — the exact FP64 bounded test still gates.

The **Priest non-overlap invariant** `|f_{i+1}| ≤ ½ ulp(f_i)` (bit form
`fl(f_i + f_{i+1}) == f_i`) is checked on every `renorm` output, oracle-independent,
with the same underflow-tail gate (`< 2⁻¹⁰⁰`) as `ff_invariant_test` (T2.2), plus a
packing check (no nonzero word after a zero word). Named cases cover inf/nan
propagation through `renorm` (its `if (isinf(c0)) return;` guard must not crash).
Device parity (Test E) re-runs `qf_two_sum` / `qf_two_prod` / `renorm_4` in a
`parallel_for`. Registered with `kokkos_ep_add_eft_test(qf_eft_test)` (contraction
OFF); the contraction-ON reporter mirror is T3.5.

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

### FF non-overlap invariant (Layer 2, Phase 2)

`ff_invariant_test.cpp` (**T2.2**) is the FF analogue of `dd_invariant_test.cpp`,
mirroring its structure verbatim (50-row op inventory, two-pass random+corpus
shape, skip-not-fail domain gating, per-op reporting, 5-op device pass, Test C
PORT_NOTES §4 regressions). The one type change: the invariant is evaluated in
**raw FP32** — `(f.hi + f.lo) == f.hi` in `float` — where the DD test uses
`double`. Still **no** `__float128` promotion (that tests the exact real sum, a
different property) and **no** `KOKKOS_EP_HAVE_QUADMATH` gate: like T1.2 it runs
unconditionally. Every op in `ff_math.hpp` that returns a `FloatFloat` (or two via
out-params for `sincos`/`sinhcosh`) has a corresponding FF entry — no DD op is
missing on the FF side. Registered with the plain `kokkos_ep_add_test` helper (no
contraction flags).

**FP32-narrower domain predicates.** Every predicate was **re-derived** from the
shipped `ff_math.hpp` guards (not copied from T1.2) and empirically confirmed to
emit **zero** internal diagnostics. FP32's exponent range is ~6× narrower than
FP64, and the port has FP32-specific hazards T1.2's FP64 code never hit. The
material tightenings:

- `exp` gates at `a.hi < 88.0` (FP32 ln-range guard in `ff_math.hpp`), not DD's
  300; `exp2`/`exp10` scale accordingly (`|a|<126` / `|a|<38`).
- **trig family** (`sin`/`cos`/`tan`/`atan`, plus `atan2` and the `tgamma`
  reflection path) carries a **lower** magnitude bound (`|x| ≥ 1e-25`, or exactly
  0): FF's `sincos` Taylor loop hits its iteration limit (`FFCSSNR`) for tiny
  nonzero arguments because `r = x/2^nq` underflows at FP32. DD's FP64 `sincos`
  never saw this, so this floor has no T1.2 counterpart.
- `atan2` additionally floors **both** operands away from `|·|<1e-18` (0 allowed):
  a subnormal-tiny operand paired with a normal one drives the internal `sincos`
  degenerate — an FF-specific tightening of DD's larger-magnitude-only gate.
- `log`-family window `[1e-34, 1e34]` (keeps `|log x| < ~78`, inside `exp`'s
  88-guard); `sinh`/`cosh` `|x|<40`, `tanh` `|x|<20`, `tgamma` `x∈[1e-3, 23)`,
  `asinh`/`acosh`/`atan` upper caps `1e18` (so `x·x` stays finite in FP32).

**Test C — PORT_NOTES §4 regressions.** The `exp` §4a cases split into **two
distinct roles** (kept explicit in the test output so readers don't conflate them):

- **79.5 / 80 / 85 — bug-regression cases.** The historical §4a bug was
  NaN-from-splitter-overflow (`exp`'s internal Dekker split did `b * 8193.0f`,
  which overflowed FP32 → NaN). These three are the **load-bearing** regression
  cases: pre-fix they returned NaN; post-fix (direct scaling) they return finite,
  invariant-clean results. No diagnostic expected.
- **88.7 / 88.72 — edge-of-saturation guard cases.** These sit **past** the shipped
  `a.hi ≥ 88` guard and do *not* re-test the §4a bug; they assert the guard fires
  sensibly at the edge — saturating to **+0** (invariant trivially holds, not-NaN)
  rather than producing garbage. Each emits one `FFEXP: argument too large` print;
  those **2** diagnostics are the *only* internal `ff_math.hpp` output in the whole
  run (Test A/B are diagnostic-clean) and are **expected and normal** — documented
  safety-guard behavior is a pass, not a report-and-stop.

`round_to_nearest_int` at 19.4999993 and the k±0.5 family; `remainder(68.379,
3.5066)` gated against `std::remainderf` (the **same-precision** FP32 oracle).

> **Finding (remainder sign).** The T2.2 prompt (following PORT_NOTES §4b) expected
> a *positive* FP32 remainder here, on the premise that `a/b ≈ 19.4999993 < 19.5`
> at FP32 → `nint=19`. That premise does **not** hold for the corpus literals: at
> FP32 `68.379f/3.5066f = 19.5000858` (**>** 19.5), so the correct `nint` is 20 and
> the correct remainder is **negative** (−1.75300026). `std::remainderf` agrees,
> and the shipped FF `remainder` reproduces it exactly. The §4b "+1.7533" text
> describes a different rounding of `a/b` than these specific literals produce. The
> test gates against `std::remainderf` (not `std::remainder`, which would compare an
> FP32 op to the FP64 answer) and passes. See the Test C comment for the full
> derivation.

> **Finding (nint literal).** `19.4999993f` rounds to **exactly** `19.5f` at FP32
> (`lo=0`), so `round_to_nearest_int` of the pure-float value returns 20 — correct.
> The historical `19`-vs-`20` distinction only appears when the full-precision value
> `19.4999993` is carried in the FF pair via the Route-A double split (`hi=19.5`,
> `lo=−7e-7`, total < 19.5), where the fixed `ffnint` (rounding `hi+lo` in FP64)
> returns 19. Test C checks both constructions.

> **Finding (underflow-tail ties → `kUnderflowTail` skip).** The bulk random+corpus
> pass initially reported "failures" for `exp`/`exp2`/`exp10` (and `device:exp`) on
> **very negative** arguments — e.g. `exp(-84.32) → hi=2.40e-37, lo=-1.121e-44`.
> These are **not** normalization defects. Each failing `lo` is a **subnormal** that
> lands *exactly* on the `½ ulp(hi)` tie point (`-1.121e-44 = -½·2⁻¹⁴⁵` for that
> `hi`), where round-to-even flips `fl(hi+lo)` off `hi` by one ulp **even though the
> mathematical non-overlap `|lo| ≤ ½ ulp(hi)` still holds**. This is systematic only
> in the FP32 denormal tail: once `|hi| < 2⁻¹⁰²`, the tie value `½ ulp(hi) = 2^(e−24)`
> is itself subnormal, so the residual `lo` is quantized straight onto the tie. The
> strict bit-exact form `fl(hi+lo)==hi` is therefore **ill-posed** there — a property
> of double-word arithmetic near underflow, universal to DD/FF/QF, not an
> `ff_math.hpp` bug (DD rarely hits it because FP64 underflows ~270 decades lower).
> `result_checkable` skips this tail via `kUnderflowTail = 2⁻¹⁰⁰` (a 4× margin above
> `2⁻¹⁰²`); the guard is output-side and general (any op), and does **not** mask a
> real overlap — a normal-range `hi` with `|lo| > ½ ulp(hi)` is still checked. With
> the guard, all ~48.8M checked inputs pass with **zero** failures.
>
> **Not FF-specific — DD has the same latent hole.** This round-to-even hole in the
> `fl(hi+lo)==hi` *evaluation* is a property of double-word arithmetic, not of FF:
> DD (and a future QF) can hit it too. `dd_invariant_test` (T1.2) simply never
> tripped it across ~50.5M inputs because FP64's exponent range is ~6× wider, so the
> denormal tail is out of reach for realistic random inputs. FP32's narrower range
> brings the tail into reach at *ordinary* op inputs (`exp` of any sufficiently
> negative argument), which is why FF surfaced it first. **Follow-up (not urgent, not
> a blocker):** give `dd_invariant_test` the same `kUnderflowTail`-style guard so it
> is not surprised by the same hole if the DD op inventory grows or a future random
> seed lands in the tail. Flagged as a cross-cutting known-lurking issue in the T2.2
> DONE block.

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

### FF property/identity (Layer 3, Phase 2)

`ff_property_test.cpp` (**T2.3**) is the FF analogue of `dd_property_test.cpp`,
mirroring its structure verbatim. The only substantive change is the **precision
scale**: FF's unit roundoff is `u = 2⁻²⁴` (vs DD's `2⁻⁵³`), so
`u² = 2⁻⁴⁸` and the statistical floor becomes
`tolerance_digits = -log10(N·u²) ≈ 8.45` at `N = 10⁶` (vs DD's ≈ 25.91).
Everything is computed at runtime from `BackendTraits<FF>::u_squared`, not
hardcoded. Group B means land around 13 digits, clearing the ~8.45 floor with
room to spare.

**Group A (7 bit-exact).** Same identities as DD: `add(a,negate(a))==0`,
`a-a==0`, `a·1==a`, `a·(-1)==negate(a)`, `abs` sign branches,
`negate(negate(a))==a`, add commutativity. Failures dump raw FP32 bit patterns
(`0x%08x` per limb). A1–A6 use full Route-A FF operands (`FloatFloat(double)`,
generally **nonzero** `lo`); A8 (add commutativity) uses **single-float**
operands (`lo==0`) — the FF analogue of DD's single-double convention: with a
nonzero `lo`, `add()`'s trailing `+a.lo+b.lo` reorders under operand swap and
would break bit-exactness. Multiply-by-±1 (A3/A4) is Dekker-domain-gated
(`split_safe_max() = FLT_MAX/8193`), and any strict-`==` mismatch whose limbs
land in the FP32 denormal tail (`< 2⁻¹⁰⁰`) is counted **skipped**, not failed
(the T2.2 round-to-even hole).

**Group B (13 tolerance).** Same identities as DD, mean-gated at ≈ 8.45. The
exp round-trips narrow their domains to respect `ff_math.hpp`'s exp guard
(`a.hi ≥ 88.0f` returns 0): B2 `exp(log(a))` on `[1e-30,1e30]`, B11 `pow(a,2)` on
`[1e-15,1e15]` (`2·ln(1e15) ≈ 69 < 88`). B3 `log(exp(a))` and B9 `exp(a)·exp(-a)`
are further narrowed to `[-69,69]` (from `[-85,85]`) per pending follow-up bug
task **B4**: exp's Taylor convergence `eps=1e-15f` is finer than FloatFloat's
`~3.55e-15` resolution, so for ~3 % of generic large-magnitude arguments exp
stalls to its iteration cap, prints `FFEXP: iteration limit`, and returns 0
(surfaced via `log()`'s internal Newton exp; results stay accurate but stdout is
spammed). Restore to `[-85,85]` once B4 lands. B8 double-angle narrows to `|a| < 3` because it
compares two *different* reduced arguments and FF's double-float argument
reduction degrades for large `|a|` (PORT_NOTES §5). B0 is the demoted multiply
commutativity.

**Test C.** Target ≥ 12 of FF's 14 digits (DD used ≥ 30 of 31): `log(e)≈1`,
`exp(log2)≈2`, `√2·√2≈2`, `log(10)≈log10` constant. C1 `|sin(π)|≈0` is softened
to a conditioning-aware floor (arg reduction near π, PORT_NOTES §5); C6
euler_gamma/digamma is skipped (no digamma op in `ff_math.hpp`).

**Device pass.** 3 Group A (`A1`, `A3`, `A5`) + 2 Group B (`B1`, `B4`) on 10⁵
inputs, with `View<float*>` limb transfer. Registered with the plain
`kokkos_ep_add_test` helper (no contraction flags — identities are
FMA-contraction agnostic). `ff_math.hpp`/`ff_complex.hpp` are **not** modified by
this layer.

## Differential accuracy (Layer 4)

Layer 4 (`dd_accuracy_test`, **T1.4**) asks the question Layers 1-3 deliberately
did not: *does `op(x)` equal the true real answer to N digits?* For **every** op
in the T1.2 inventory (~50 rows), it widens the device result to `__float128` and
compares against a quadmath oracle evaluated on the SAME input, scoring each
element in **digits of accuracy** `digits = -log10(rel_err)` (capped at DD's 31 =
`-log10(u²)`, u = 2⁻⁵³). Two passes per op — **10⁶ random** (ranges taken
verbatim from the T1.2 domain predicates) plus a **corpus** pass (the PORT_NOTES
§3/§4 named accessor where one exists, e.g. `exp_overflow`, `trig_near_pi`,
otherwise the generic bundler) — combined into one (min, mean, n) per op.

**Fail-gates on the MEAN**, not the min, against a single uniform
`tolerance_digits = -log10(N·u²) ≈ 25.91` at N = 10⁶ (no per-op tolerance
overrides — that would defeat the point of a differential-accuracy gate). Ops in
the shared PORT_NOTES §5 conditioning registry (`lookup_expected_min_drop`)
report **EXPECTED-MIN-DROP: OK** when the mean clears tolerance and the low min is
sanctioned (near-cancellation, derivative → ∞ near |a|=1, arg-reduction near ±π,
output-denormal `exp`). Oracle subtleties handled: ties-to-even round-family →
`nearbyint` oracle (not `round`); `exp10` → `pow(10, x)` (no `__float128`
overload). The whole file is `#ifdef KOKKOS_EP_HAVE_QUADMATH` and runtime-SKIPs
(77) otherwise.

`dd_accuracy_test` ships **`(DONE, RED)`**: it flags three real `dd_math.hpp`
accuracy defects (`tgamma` mean ≈ 14.56 — FP64 Lanczos coefficients; `erfc` ≈
19.50 — `1−erf` cancellation; `erf` ≈ 24.64 — large-|z| asymptotic branch) and
fails on them. The red is the point — it is the durable regression gate for the
follow-up bug tasks B1/B2/B3. Per rule 4 the surfacing test reports; it does not
patch the library.

### FF differential accuracy (Layer 4, Phase 2)

`ff_accuracy_test.cpp` (**T2.4**) is the FF analogue of `dd_accuracy_test.cpp`,
mirroring its structure verbatim. The substantive changes are the **precision
scale** and the **FP32-narrower op domains**:

- **Scale.** FF's unit roundoff is `u = 2⁻²⁴` (vs DD's `2⁻⁵³`), so `u² = 2⁻⁴⁸`,
  digits are capped at **14** (`BackendTraits<FF>::max_digits`), and the
  statistical floor becomes `tolerance_digits = -log10(N·u²) ≈ 8.45` at N = 10⁶
  (vs DD's ≈ 25.91) — computed at runtime from `BackendTraits<FF>::u_squared`, not
  hardcoded. Expected mean per PORT_NOTES: 13.3-14.0.
- **Domains.** Random ranges and domain predicates are taken **verbatim from the
  T2.2 inventory** (`ff_invariant_test.cpp`), not re-derived: `exp` guards at
  `a.hi ≥ 88` (not DD's 300); trig carries the FP32 tiny-argument lower bound
  (`|x| ≥ 1e-25`, else 0, to dodge the sincos iteration-limit hazard);
  `sinh`/`cosh` cap at `|x| < 40`, `tanh` at `|x| < 20`; the log family window is
  `[1e-34, 1e34]`; `erf`/`erfc` use `[-6, 6]` (FF saturates to ±1 past |z|=6);
  `tgamma` uses `[1e-3, 23)`. The corpus pass uses the FP32 accessors
  (`corpus::unary<float>` / `<float>` named accessors).

Same op set as T1.4 (~50 rows: arithmetic, transcendentals, roots, comparisons,
two-output `sincos`/`sinhcosh`, ternary `fma`, integer-scalar `pow_int`), the same
oracle subtleties (`nearbyint` for the ties-to-even round-family, `pow(10,x)` for
`exp10`), the same MEAN fail-gate with EXPECTED-MIN-DROP for the **shared**
PORT_NOTES §5 registry (conditioning is a property of the algorithm, not the
width, so DD and FF read the same table). Registered with the plain
`kokkos_ep_add_test` helper (no contraction flags — not an EFT test), mirroring
`dd_accuracy_test`. Per rule 4, `ff_math.hpp` / `ff_complex.hpp` are **not**
modified: any op whose mean falls below tolerance is REPORTED (op, pass, offending
input, digit count) and fails; it is not patched or xfailed.

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

### FF FMA-contraction guard (Layer 5, Phase 2)

`ff_fma_guard_test.cpp` (**T2.5**) is the FF analogue of `dd_fma_guard_test.cpp`,
mirroring its structure verbatim: one source compiled into two targets under
opposite contraction postures, a contraction-immune oracle, a `twoSum` **control**
(no mul-then-± adjacency → must stay exact both ways), host + device passes, OFF
gates / ON reports with a committed baseline. It reuses the **same** CMake helpers
as the DD guard:

```cmake
kokkos_ep_add_eft_test(ff_fma_guard_test)              # -> ff_fma_guard_test              (OFF, gates)
kokkos_ep_add_eft_test_contract_on(ff_fma_guard_test)  # -> ff_fma_guard_test_contract_on  (ON, reports)
```

The `_contract_on` helper derives the per-test baseline path
(`ff_fma_guard_baseline.txt`) from the target name, so the DD and FF guards share
one helper with no duplication.

The FF Dekker `twoProduct` is the **same algorithm** at FP32: splitter `8193.0f`
(2¹³+1), hi/lo split, the cross-term subtraction `a1*b1 - p`. The contraction
hazard is identical — a fused `fma(a1, b1, -p)` computes the tail with
full-precision residuals the algorithm's rounded-intermediate algebra does not
assume, silently breaking the error term.

**Two divergences from the DD shape, both reported (see the T2.5 DONE block):**

- **FP64 oracle, no quadmath gate.** Like `ff_eft_test` (T2.1), ground truth is
  plain **FP64**, not `__float128`: the exact FP32 product needs ≤48 bits, which
  fits FP64's 53-bit mantissa, so the reference `(p, e)` decomposition is
  *provably* exact — a stronger oracle than DD's quadmath, needing no external
  library. Consequently both variants run **unconditionally** (no
  `KOKKOS_EP_HAVE_QUADMATH` gate, no runtime SKIP-77), unlike the DD guard which
  SKIPs without LIBQUADMATH.
- **`twoSum` control gets an FP32-specific oracle-faithfulness guard.** The DD
  guard reuses its `twoProduct` input list for the `twoSum` control unchanged;
  that is safe at FP64 scale but **not** at FP32. A pair like `(FLT_MIN = 2⁻¹²⁶,
  2²⁴)` has an in-range *product* (2⁻¹⁰², admitted by the `twoProduct` domain) yet
  an exact *sum* spanning 174 bits — far beyond the FP64 oracle's 53. There the
  FP32 `twoSum` is **correct** (`hi = 2²⁴`, `lo = 2⁻¹²⁶`); it is the FP64
  *decomposition* that collapses the tiny tail, so the exact `lo` looks like a
  false "mismatch". The control therefore skips pairs the oracle cannot witness
  (`sum_oracle_faithful`: the FP64 `twoSum` error term is zero ⇔ the double sum is
  exact), excluding **only** wide-exponent sums, never a pair where FP32 `twoSum`
  is actually wrong. `twoProduct` needs no such guard — its ≤48-bit product is
  always oracle-faithful (confirmed: 0 mismatches).

**Observed on this toolchain (GCC 13.3.0, baseline x86-64, no `-mfma`):** the FF
Dekker `twoProduct` is bit-exact over **220,410** in-domain checks (host + device)
under **both** postures — **`F = 0`**, the *same* outcome T1.5 recorded for DD.
GCC emits plain mul+sub and contracts nothing on this ISA; `-ffp-contract=off` is
belt+suspenders here, and the ON reporter's baseline (`0`) arms drift detection for
a future toolchain where that stops being true. Because `F = 0`, the ON variant is
a **reporter** (exits 0), not a `WILL_FAIL` test — matching
`dd_fma_guard_test_contract_on` exactly (the guard's realness is proven by
sensitivity, not by contraction on this compiler; T1.5 verified a deliberately-
broken `twoProduct` flags ~95 % of checks). **Scope:** the Dekker `twoProduct`
only; no complex, no other ops; `ff_math.hpp` NOT modified.

## End-to-end cancellation kernels (Layer 6)

Layer 6 (`dd_e2e_test`, **T1.6**) is the payoff the end user actually cares about.
Layers 1-5 validated a backend's atoms (EFT), structure (non-overlap), identities,
per-op accuracy, and FMA-contraction posture — all machinery. Layer 6 asks: on
classic **cancellation-hostile** problems that the base scalar mangles, does the
double-word backend deliver its advertised digits? Four kernels, each with a known
higher-precision or closed-form oracle:

- **K1:** `√(x²+1) − x` — catastrophic cancellation at large `x`
  (`√(x²+1) ≈ x`, the answer `~1/(2x)` lives in the surviving low bits).
- **K2:** `Σ 1/k²`, k=1..10⁶ — Basel problem, closed form `π²/6`.
- **K3:** Machin's `π = 16·atan(1/5) − 4·atan(1/239)` — transcendental composition.
- **K4:** `Σ (−1)^(k+1)/k`, k=1..10⁶ — alternating harmonic, closed form `ln 2`.

**Two-oracle strategy (K2, K4).** Each finite sum is scored twice. The
sum-vs-**quadmath-partial-sum** comparison (identical N, order, terms) carries the
**arithmetic-precision** claim — it isolates accumulation quality from truncation.
The sum-vs-**closed-form** comparison (K2 vs π²/6, K4 vs ln 2) is a
**truncation-limited sanity check**, gated at `truncation_floor − 1`; at N=10⁶ the
floor is ~6 digits (the Basel tail and the alternating-series error are both ≈ 1/N).

**K1 deviation (documented, both backends).** The literal spec named naive
`√(x²+1) − x` as the DUT expecting full precision; that expectation is numerically
false and **not** a library defect — cancellation loses ~2·log₁₀(x) digits
regardless of arithmetic (Higham §1.7). So K1 ships as a **gated** stable form
`1/(√(x²+1) + x)` (algebraically cancellation-free) plus a **reported, not gated**
naive form (backend vs base scalar, per magnitude) that demonstrates the
extra-word lift under the hostile algorithm.

Both kernels are **host-side** (inherently serial reductions/recurrences), whole
file `#ifdef KOKKOS_EP_HAVE_QUADMATH` (SKIP 77 without quadmath), and neither
modifies the backend math header (rule 4). Registered with the plain
`kokkos_ep_add_test` helper (not an EFT test).

**DD (T1.6).** Gate `mean_digits ≥ 28.0` (= DD's cap 31 − 3 headroom). K1 uses
x ∈ {1e6, 1e10, 1e15} and reports naive DD vs **FP64** (DD's base scalar). Measured
means: `K1_stable` 31.00 (capped), `K2` 29.48, `K3` 28.09, `K4` 29.56 — all PASS.

### FF end-to-end cancellation (Layer 6, Phase 2)

`ff_cancellation_test.cpp` (**T2.5**) is the FF analogue of `dd_e2e_test.cpp`,
mirroring its structure verbatim (same four kernels, same two-oracle strategy, same
K1 gated-stable + reported-naive shape, host-side, quadmath-gated, `ff_math.hpp`
untouched). The substantive changes are the **precision scale** and two
**FP32-forced K1 deviations**, both derived (not fabricated) and reported in-source:

- **Gate.** `mean_digits ≥ 11.0`, derived by the **same "cap − 3" formula** T1.6
  used: FF's harness cap is `BackendTraits<FF>::max_digits = 14` (u² = 2⁻⁴⁸ ≈ 14.45
  decimal digits), so `14 − 3 = 11.0`. Computed from `max_digits` at compile time,
  not hardcoded.
- **K1 baseline = FP32, not FP64.** T1.6 compared naive-DD against naive-FP64 (DD's
  1-word base). The faithful FF mirror compares naive-FF against naive-**FP32** (FF's
  1-word base). Comparing FF against FP64 would be dishonest — FP64 (~16 digits) is
  *wider* than FF (~14), so it would "win" the naive contest while saying nothing
  about FF. FF's advantage is over its own base scalar, exactly as DD's is over FP64.
- **K1 magnitudes {1e2, 1e4, 1e6}, not {1e6, 1e10, 1e15}.** The cancellation
  gradient lives ~3 decades lower for FF: plain FP32 loses the `+1` in `x²+1` once
  `x² > 2²⁴` (x ≳ 4100) and FF loses it once `x²` exceeds FF's ~14-digit reach
  (x ≳ 1e7). At T1.6's magnitudes both naive forms would read 0 at the upper two x —
  no gradient. At {1e2,1e4,1e6} the FP32→FF lift is visible across the whole sweep.

K2/K4 keep N = 10⁶: at that N the smallest term (1/N = 1e-6 for K4, 1e-12 for K2)
stays well above FF's running-sum resolution, so no term stalls into the precision
floor and the arithmetic-precision comparison is well-posed (the iteration-bound
concern the plan flags for FP32 does not bite here). Per-kernel measured results
are printed by the test and recorded in the T2.5 DONE block. Registered with the
plain `kokkos_ep_add_test` helper (no contraction flags — see the CMake comment on
why K1's naive mul-then-sub adjacency is not a gated-path hazard).

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
