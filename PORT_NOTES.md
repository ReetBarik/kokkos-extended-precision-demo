# Porting notes: DD → FF backend

This branch (`fffunKokkos`) was forked from `ddfunKokkos` and the FF (float-float)
backend was meant to be a mechanical translation of the DD (double-double) backend:
swap `double`→`float`, `ddouble`→`ffloat`, namespace, function prefixes, and the
Dekker splitter constant. That covers maybe 80% of the work. This file documents
the other 20% — issues that surfaced only after the demos ran and the accuracy
table looked wrong.

The point isn't "DD has bugs"; it's that some DD techniques rely on FP64 having
enough dynamic range to absorb sloppiness that FP32 can't. Each lesson below
explains a thing that broke, what fixed it, and why it didn't show up in DD.

---

## 1. Demo input narrowing — every op looked algorithmically broken

**Symptom**: first run showed every operation, including pure data-shuffling
ops like `abs`, `copysign`, `fmax`, `fmin`, capped at ~7.8 digits of accuracy.
Pure data ops *cannot* lose precision, so this had to be input-side.

**Cause**: the demo populated FF inputs with
```cpp
maff(i) = ff::ffloat((float)ha[i]);
```
The explicit `(float)` cast routes through the `ffloat(float)` constructor,
which sets `lo = 0`. So the FF input only carried the top 24 bits of the FP64
value `ha[i]`, while the quadmath reference saw the full 53 bits. Best-case
accuracy was bounded by `log10(2²⁴) ≈ 7.2 digits`, regardless of FF
arithmetic quality.

**Fix**: drop the cast and use the `ffloat(double)` constructor, which does the
Route-A split (`hi = (float)x`, `lo = (float)(x − (double)hi)`) and faithfully
encodes the FP64 value to ~14 digits.

**Why DD didn't see this**: the equivalent `ddouble(double)` constructor sets
`lo = 0`, but DD `hi` is FP64 — the same precision as the input — so no
information is lost.

---

## 2. Build environment

### 2a. Kokkos 5.1 dropped C++17

Both `CMakeLists.txt` and `scripts/build_with_kokkos.sh` pinned
`CMAKE_CXX_STANDARD=17`. Kokkos 5.1 errors out at configure time:
> Kokkos requires C++20 or newer but requested 17!

Bumped both to C++20. No code changes needed — straight C++17 is a subset of
C++20 for the constructs this codebase uses.

### 2b. Silent build-script failures

The Kokkos install step is wrapped in
`{ git clone; cmake; make install; ... } 1>stdout.log 2>stderr.log`. The
redirect swallowed cmake errors, the script kept running, and the project's
own cmake then failed with an unhelpful `find_package(Kokkos) not found`.

**Fix**: after the brace block, check `[ -f $TARGET_DIR/kokkos/setup.sh ]`
and bail with a pointer to the log files if it's missing.

---

## 3. Precision fixes that mattered

These were uncovered by reading the accuracy table critically — looking at
*which* ops had bad mins and asking "what's the algorithmic mechanism?"

### 3a. `sincos`: track both sin and cos through the doublings

**Original DD-style algorithm**: compute `sin(s3/2^nq)` via Taylor, then double
`cos` through the loop using `cos(2x) = 1 − 2sin²(x)`, then recover `sin(s3)`
at the end via `sin = sqrt(1 − cos²)`.

The recovery step loses relative precision exactly when `|sin|` is small —
i.e., near multiples of π — which is precisely when `sin`'s relative error
matters most. `tan(7.5)` and `asin(7.4)` digits in the original FF run were
the symptom.

**Fix**: track sin *and* cos through the doublings using
`sin(2x) = 2·sin(x)·cos(x)`, `cos(2x) = cos²(x) − sin²(x)`. No final sqrt.

**Cost**: ~25–30% slowdown for `sin`/`cos`/`tan` because each doubling now does
~4 multiplications instead of ~1.

### 3b. `sinh`/`cosh` near zero: Taylor branch

For small `a`, `(eᵃ − e⁻ᵃ)/2` cancels: both exponentials are near 1, the
leading bits subtract away. Same shape as `exp(a) − 1` near zero, which the
library already handled via `expm1`.

**Fix**: when `|a| < 0.5`, compute the Taylor series for both `sinh` and
`cosh` directly (no exp call). For `|a| ≥ 0.5` the exponentials are
sufficiently far apart that the original formula is fine.

### 3c. `atanh`: Taylor, not log

First attempted fix was `0.5·(log1p(a) − log1p(−a))`, on the theory that
`log1p` is well-conditioned for small arguments. It actually regressed both
accuracy *and* runtime (each call now did two log evaluations).

Real lesson: `log1p(small) = log(1 + small)` still has `log` evaluated near 1,
which loses precision in the Newton iteration's residual computation. The
right fix is to bypass `log` entirely.

**Fix**: for `|a| < 0.5`, use `atanh(a) = a + a³/3 + a⁵/5 + ...` (all positive
terms, no cancellation). For `|a| ≥ 0.5`, fall back to the original
`0.5·log((1+a)/(1−a))` — the ratio is far from 1, no cancellation in `log`.

Net result: faster *and* more accurate.

---

## 4. Outright bugs

**§4a–§4b — found by the probe.** The accuracy table had two ops showing
`min digits = 0.0` (i.e. at least one sample produced a wrong result).
`scripts/probe_op.cpp` re-runs the demo's inputs on the host and prints the
worst-accuracy elements with bit patterns, which made both bugs obvious.

**§4c–§4h — found by the test suite.** These were added later, by the
post-Phase-3 library-side fix tasks (B1–B8 in `docs/TEST_SUITE_PLAN.md`), which
the accuracy and property suites surfaced rather than the probe. They cover the
DD backend as well as FF: the DD entries live here, next to their FF siblings,
because in each case the lesson is a shared-series or shared-identity one rather
than a format-specific one. (QD → QF port notes live separately in
`docs/PORT_NOTES_QF.md`.)

### 4a. `exp` NaN at large inputs

**Symptom**: every `min=0.0` element had input `a > 79.4` and FF output `NaN`.

**Cause**: the final scaling step in `exp` used
```cpp
return ffmulf(s3, ldexpf(1.0f, nz));
```
For input `a ≈ 80`, `nz = 115`. Inside `ffmulf`, the Dekker splitter computes
`b * 8193.0f` where `b = 2¹¹⁵`. That product is `2¹²⁸ ≈ FP32_MAX`. For
`nz ≥ 116` the splitter overflows to `inf`, then `inf − inf = NaN` poisons
everything downstream.

**Fix**: power-of-2 multiplication is exact in FP32 (no rounding) and doesn't
need Dekker splitting. Replace with direct scaling:
```cpp
float pow2 = ldexpf(1.0f, nz);
return ffloat(s3.hi * pow2, s3.lo * pow2);
```

**Why DD didn't see this**: DD's splitter is `134217729.0` and the FP64
exponent range easily absorbs `134217729 · 2¹⁵⁰⁰` without overflow. FP32's
6× narrower exponent range puts the splitter overflow squarely inside
`exp`'s normal operating range.

### 4b. `ffnint` off-by-one near half-integers

**Symptom**: `remainder(68.379…, 3.5066…)` returned `−1.7533` when the
reference said `+1.7533`. Caused by `ffnint(19.49999930…)` returning `20`
instead of `19`.

**Cause**: the DD-style nint trick adds a magic constant `2^(2p−1)` (where
`p` is per-component precision), then subtracts it. The addition forces the
input's fractional bits to round to integer. For DD, the constant is `2¹⁰⁵`
and the trick is well-conditioned because FP64 has 53-bit mantissa to absorb
small-integer additions cleanly. For FF, the constant is `2⁴⁷` — but FP32 has
only 24-bit mantissa, so the ULP at 2⁴⁷ is 2²⁴ = ~17 million, vastly larger
than any expected input. The FF `lo` component was supposed to rescue the
precision; for inputs near half-integers it doesn't.

**Fix**: do the rounding in FP64. FF values are bounded by 2⁴⁸ and fit
exactly in FP64's 53-bit mantissa. The FP64 magic-constant trick (using
2⁵²) is the standard, well-known one and is bullet-proof in this regime.
```cpp
double total = (double)a.hi + (double)a.lo;
const double T52 = ldexp(1.0, 52);
double rounded = (total > 0.0) ? (total + T52) - T52 : (total - T52) + T52;
return ffloat((float)rounded, (float)(rounded - (double)(float)rounded));
```

This also benefits `floor`, `ceil`, `round`, `trunc`, `fmod`, and the
argument reduction in `sincos` — all of them call `ffnint`.

### 4c. `tgamma` Lanczos coefficients at FF precision: an `f` suffix cost half the mantissa

tgamma Lanczos coefficients at FF precision (B7). The DD->FF port suffixed the
g=7 coefficient literals with `f`, truncating each to FP32 (~7 digits) and
capping tgamma. Fix: store as `double`; FloatFloat(double) splits to a full FF
pair (double's 53 bits > FF's 48, so FF-exact). sqrt(2*pi) leading factor
promoted identically. Structure unchanged. The DD sibling (B1) has the analogous
defect but caps at ~15 digits, not ~7, since DD has mantissa headroom above the
double coefficients -- so B1 needs true DD-precision coefficients, whereas FF
only needs double.

Cross-ref: TEST_SUITE_PLAN.md §B7 DONE (commit 926e056).

### 4d. `divide`'s Dekker splitter: the same overflow as §4a, at a second site

**Symptom**: `log(x)` for `x ≳ e^79.7 ≈ 1.5e34` returned NaN. The NaN fed back
into `exp()` on the next Newton step, and because NaN comparisons always
evaluate false, the Taylor convergence check never broke — the loop stalled at
its 60-iteration cap and printed `FFEXP: iteration limit`.

**Cause**: `divide()` extracts the divisor's high half with a Dekker split that
computes `conb = b.hi * split` (`split = 8193.0f = 2^13+1`). For
`|b.hi| > FLT_MAX / (split + 1) ≈ 4.15e34` that product overflows to `±inf`,
and then `b1 = conb - (conb - b.hi) = inf - inf = NaN` corrupts the split.
`log()`'s Newton iteration divides by `exp(b) ≈ a`, which crosses that band for
`x ≳ 1.5e34`.

**Fix**: scaled splitter. When `|b.hi|` is in the overflow-hazard band,
pre-scale the divisor down by the exact power of two `s = 2^-64` (power-of-2
multiplication does not round, so `b`'s full FF precision survives), run the
*unchanged* Dekker split, then unscale. `2^-64` leaves ~15 orders of headroom:
the largest `|b.hi| ≈ FLT_MAX = 2^128` maps to `2^64`, and
`2^64 · split ≈ 2^77 ≪ FLT_MAX`. The non-overflow path (`s = 1.0f`) is
bit-identical to the prior code.

**Watch the direction of the unscale.** `q = a / (b·s) = (a/b) / s`, so
recovering `a/b` means **multiplying** `q` by `s`, not by `1/s` — the latter
yields `(a/b)/s²`. This is the easy sign error in every scaled-splitter fix.

**Why DD didn't see this**: DD's splitter is `134217729.0` (2^27+1) against
FP64's ~1.8e308 range, so the hazard band sits far outside any operand a real
computation produces. Same bug class as §4a; different site (divide's splitter,
not exp's final scaling). §4a was deliberately left unmodified.

Cross-ref: TEST_SUITE_PLAN.md §B8 DONE (commit b2cff7d). Composed at
consolidation time from the B8 DONE block and task commit — no verbatim draft
was parked at close.

### 4e. `erf` at FP32: separately-grown numerator and denominator overflow before they divide

**Symptom**: FF `erf` returned outright NaN across ~[1.9, 6] — a smooth,
well-conditioned range with nothing special about it (mean 3.94, min 0.00).

**Cause, two compounding defects.** (a) The DD port accumulates each Taylor
term from a separately-grown numerator `t2 = 2^k z^{2k+1}` and denominator
`t3 = (2k+1)!!`. At FP64 both intermediates stay finite; at FP32 (max ~3.4e38)
both overflow around `k ~ 26–31` for `|z|` in [2, 4), so `t2/t3 → inf/inf = NaN`
before the convergence test ever fires. (b) The asymptotic branch's erfc series
is *divergent*, so its relative-eps test can never trip; the loop ran to its
fixed `k=60` cap, where `(2k−1)!!` overflows FP32 to the same NaN.

**Fix**: rewrite both branches around a term recurrence — Taylor
`term_k = term_{k−1} · 2z² / (2k+1)`, asymptotic
`term_k = term_{k−1} · −(2k−1) / (2z²)` — so every intermediate stays `O(term)`
and FP32 overflow becomes structurally impossible for `|z| ≤ 6` (the Taylor sum
is bounded by `(√π/2)·e^{z²}`, ~3.8e15 at `|z|=6`). Add optimal truncation to
the asymptotic branch: stop as soon as `|term_k| > |term_{k−1}|`, the
smallest-term criterion. Move the switchover 4.0 → 3.5. Convergence `eps`
1e-15f → 1e-14f — roughly FF's 2^-46 relative resolution; anything finer can
never fire, which is the B4 lesson carried forward.

**General lesson**: a term ratio is not merely faster than recomputing a
numerator and a denominator each iteration — at a narrow exponent range it is
the only form that survives. Wherever a DD series grows two large intermediates
and divides them, the FF port needs the recurrence, whether or not the DD
original showed any symptom.

**Cross-format note**: the DD sibling (§4g) carries the same divergent-series
and iteration-cap defects, but FP64's headroom rendered them as a gently sagging
mean instead of a NaN — which is why FF surfaced this family first.

Cross-ref: TEST_SUITE_PLAN.md §B5 DONE (commit 1013cb1). Composed at
consolidation time from the B5 DONE block and task commit — no verbatim draft
was parked at close.

### 4f. `tgamma` at DD precision: the coefficient fix that worked for FF is not enough

tgamma at DD precision (B1). Promoting the g=7 Lanczos coefficients from
double to DD -- the fix that worked for FF in B7 -- does NOT reach DD
precision: g=7/n=9 has an intrinsic truncation ceiling of ~13 digits at
large a, verified against 25-digit-exact coefficients. The order must rise
too. Shipped g=14/N=17 with coefficients derived for the partial-fraction
form. Note that published high-order Lanczos tables (Boost lanczos24m113)
are for the RATIONAL form and their g does not transfer: in the
partial-fraction form max|c_k| ~ 10^(g/2) against an O(1) sum, so high g
loses to cancellation what it gains in truncation, with an interior optimum
near g=14. Coefficients are regenerable via
scripts/gen_dd_lanczos_coeffs.py rather than transcribed.

Cross-ref: TEST_SUITE_PLAN.md §B1 DONE (commit 9c91b16).

### 4g. `erf`: a truncated series that looks like a bad expansion

**Symptom**: DD `erf` scored ~30 digits up to |z| = 5, then fell off smoothly
to 3.2 digits by |z| = 8.5, where it jumped back to full precision. FF `erf`
(B5) returned outright NaN over a comparable band.

**Cause**: the DDFUN port's `erf` has a Taylor branch and an asymptotic
branch, and the Taylor series needs `k ~ z^2 + 50` terms. Both ports capped
the loop well below that (DD 100, FF 60). DD therefore returned a truncated
partial sum — smoothly wrong, no NaN, no warning — while FP32's narrower
exponent range made FF's intermediates overflow first and NaN loudly.

The second half is a port artifact in both: the asymptotic branch's guard
(`|z| < 9.0` in DD) sits *above* the saturation cutoff (`|z| > 8.5`), so the
branch is unreachable. Dead code hides its own bugs — DD's asymptotic branch
also lacked optimal truncation, which would have shown up immediately had it
ever run.

**Fix**: switch over at |z| = 6 (where the asymptotic expansion's
smallest-term floor first clears u^2), raise the Taylor cap past the measured
requirement, and truncate the divergent asymptotic series at its smallest
term.

**Cross-format note**: this is the one place where FP64's headroom was a
*liability* for diagnosis. The same defect that FF surfaced as NaN in T2.4, DD
hid as a gently sagging mean for two more task cycles. Where DD and FF share a
series loop, check the iteration cap against the term count the series
actually needs, not against the format's exponent range.

Cross-ref: TEST_SUITE_PLAN.md §B3 DONE (commit 2bb18e3).

### 4h. `erfc`: when the identity is the bug

**Symptom**: DD `erfc` lost accuracy smoothly from ~30 digits near 0 to 16 by
z = 5.5, sat on a flat 16-digit shelf to z = 8.5, then returned exactly 0.

**Cause**: the port defines `erfc(z) = 1 - erf(z)`, which is mathematically
exact and numerically hopeless. `erf` is accurate to u^2 *relative to 1*, so
the difference carries an absolute error ~u^2 and `erfc` loses
log10(1/erfc(z)) digits — by construction, at every precision. No amount of
work on `erf` fixes it. B3 demonstrated the ceiling: making `erf`'s
asymptotic branch live routed `erfc` through `erf`'s lo word, which lifted
the mean 19.50 -> 24.87 and then stalled at exactly the 53 bits a `double`
lo word can carry.

**Fix**: compute `erfc` directly from the asymptotic series wherever that
series is worth more than the subtract — here z >= 6.5 — and never form
`1 - erf` there.

**Cross-format note**: two DDFUN-port limits only become visible once a
function's output range is extended past ~1e-130. `exp` carries a hard
`|arg| >= 300 -> 0` guard that has nothing to do with IEEE range (e^-300 is
1e-131, eight decades above DBL_MIN), and the FMA-free Dekker `two_prod`
splitter overflows on operands above DBL_MAX/(2^27+1) ~ 1.3e300, so
`divide(x, huge)` returns NaN rather than 0. Both are invisible to any test
whose sampling domain keeps results near 1. Prefer `multiply(x, exp(-a))` to
`divide(x, exp(a))` in extended-precision special functions: it dodges both.

**General**: when a special function is defined as an identity over another
one, check the identity's conditioning before tuning the callee. `1 - erf`,
`1 - cos`, `log(1 + x)`, `exp(x) - 1` are the same trap; three of those have
dedicated library entry points for exactly this reason, and `erfc` is the
fourth.

Cross-ref: TEST_SUITE_PLAN.md §B2 DONE (commit ebed8c7).

### 4i. `erfc` at FP32: the same identity bug, but the shelf ends in a cliff

**Symptom**: FF `erfc` scored exactly 0 digits — returning +0 against a nonzero
true value — at every z >= 6.02, and ~7.8 digits on a noisy plateau below that.

**Cause**: `erfc(z) = 1 - erf(z)`, the §4h defect, at FP32. What is new is the
shape. §4h's DD version degrades onto a flat shelf because `erf`'s lo word is a
53-bit channel that keeps carrying erfc's value right up to DD's |z| = 8.5
saturation. FF's lo word is a 24-bit channel and FF's `erf` saturates at
|z| = 6.0, barely above its own 3.5 Taylor->asymptotic switchover. So FF gets a
short, noisy 24-bit shelf over [3.5, 6.0] (per-float mean 7.80 digits) and then
a hard cliff to zero — not the long graceful shelf DD showed. The mean-gated
accuracy row hid this: the cliff sits at the edge of the row's uniform(-6, 6)
sampling domain, so only the corpus ever reached it.

**Fix**: identical in shape to §4h — call the asymptotic series directly above
a threshold — with three FP32-specific differences.

1. *The scaling must be `multiply(x, exp(-z^2))`.* Not for §4h's reason. FF
   `exp` does have a hard `|arg| >= 88` guard, but unlike DD's arbitrary 300 it
   really is FP32's finite range, so it is not itself the surprise. The
   surprise is that `divide(sum, multiply(sqrt(pi), exp(z2)))` breaks EARLIER
   than the guard, at z = 8.93, inside `multiply`'s Dekker splitter — the §4d
   bug at a site B8 did not scale. B8 fixed `divide`'s splitter only. Any
   extended-range special function that forms a large FF intermediate and
   multiplies it is still exposed.

2. *Derive the threshold by exhaustive float enumeration, not on a grid.* The
   fallback's accuracy at any z is roundoff luck — it depends on where erfc(z)
   happens to land in erf's 24-bit lo word — so its error curve is scatter, not
   a band. A 0.00005-spaced grid over [4.85, 6.6] reports the cut is clean at
   4.9; enumerating all 1.26M representable floats in [5.4, 6.001] finds a
   regressor at 5.6338043 that the grid stepped over. At FP32 the whole
   interesting range is only a few million floats — enumerate them.

3. *The pointwise-no-regressions rule costs something here.* At DD it was free
   (flat mean across the candidate window). At FF the row mean rises
   monotonically as the cut drops, so honouring the rule at 5.75 gives up ~0.40
   digit of mean against the mean-optimal ~4.0. Worth stating explicitly so the
   next person knows the trade was made on purpose.

**Structural floor, not fixable**: `erfc`'s reported MIN stays at -0.00 after
the fix, for the same reason DD's does (§4h cross-ref). The corpus feeds z =
10.5, 19.5, 79.5 ... 100.5, where true erfc runs from 7e-50 down to ~1e-4389.
FP32's smallest subnormal is 1.4e-45. `+0` is the only representable answer and
the `__float128` oracle scores it as a total loss. Read the far-tail MEAN, or a
windowed min over the representable range, not the row min.

Cross-ref: TEST_SUITE_PLAN.md §B6 DONE (commit 243c302).

---

## 5. What is *not* fixable

Not every low-min in the accuracy table is a bug. Some are inherent to the
operation's condition number, and no algorithm operating in fixed precision
can do better. Worth listing so the next person doesn't waste time chasing
them:

- **`sub`, `fdim`, `fma`**: random pairs occasionally cancel (e.g.,
  `a − b` with `a ≈ b`); result loses one digit per matched leading digit.
  Same effect happens in FP64 — it's just hidden by the 14-digit display cap.
- **`asin`, `acos` near `|a| = 1`**: derivative is `1/√(1−a²)`, which goes
  to infinity. A tiny error in input becomes a huge error in output. Pure
  conditioning.
- **`atanh` near `|a| = 1`**: similar; `1/(1−a²)` blows up.
- **`remainder` near a multiple of `b`**: result `a − b·nint(a/b)` is the
  difference of two values of similar magnitude, so its relative precision
  is bounded by `eps · |a| / |result|`. Unbounded as `|result| → 0`.
- **`exp` at output denormal range**: when `e^a` lands near FP32's smallest
  normal (~1.18e-38), the FF `lo` component falls into FP32 denormal range
  and loses bits. Caps `exp` accuracy at ~10 digits for `a ≈ −80`.
- **`sin`/`cos` near `±π`**: even with the joint sin/cos doubling, the
  intermediate `cos(s3/2)` near π/2 carries an absolute error of ~1e-14,
  which becomes ~6 digits relative when multiplied into a small final
  `sin`. Fixing this requires triple-float π for argument reduction —
  substantially invasive (~250 LOC of careful work).

The honest reading of any FF benchmark is the **mean** column, not the min.
Means sit at 13.3–14.0 across all 39 real and 24 complex ops, which is the
expected ceiling for a 48-bit mantissa against a 14-digit display cap.

---

## Tools

- `scripts/gen_ff_constants.cpp` — Route A constants generator.
  Run once; paste output into `ff_math.hpp`. Re-run only if the Bailey DD
  constants change (they don't).
- `scripts/test_ffmul.cpp` — standalone validator for the FF multiplication
  primitive. Run after any change to `ffmul` or the Dekker splitter.
  `ffmulff` should be bit-exact; `ffmul` should peak at ~2 ulp at FF
  resolution.
- `scripts/probe_op.cpp` — debugger for accuracy outliers. Mirrors the
  demo's RNG, runs FF on host, prints the worst-K elements with input bit
  patterns. Add new ops to it as needed.
