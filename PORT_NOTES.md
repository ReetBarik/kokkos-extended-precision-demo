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

## 4. Two outright bugs the probe found

The accuracy table had two ops showing `min digits = 0.0` (i.e. at least one
sample produced a wrong result). `scripts/probe_op.cpp` re-runs the demo's
inputs on the host and prints the worst-accuracy elements with bit patterns,
which made both bugs obvious.

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
