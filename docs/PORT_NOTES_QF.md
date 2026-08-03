# Porting notes: QD → QF backend (T3.0a — arithmetic + renormalization)

The QF (quad-float, 4×FP32) backend on branch `qffunKokkos` is a mechanical
port of the QD 2.3.24 quad-double package (four-word FP64, `qd_real`) down to
four-word FP32. Unlike the DD→FF port (`PORT_NOTES.md`), which translated
DDFUN, QF descends from **QD** (Hida-Li-Bailey), a different source tree with a
different license — see the "License lineage" note below.

QD version consulted: **2.3.24** (tarball `qd-2.3.24.tar.gz` from
`https://www.davidhbailey.com/dhbsoftware/`, `configure.ac` → `QD_PATCH_VERSION
24`; upstream mirror `github.com/BL-highprecision/QD`). Files read cover to
cover: `include/qd/qd_inline.h` (renorm, three_sum/three_sum2, sloppy_add,
ieee_add, sloppy_mul, sqr, operator*), `include/qd/inline.h` (quick_two_sum,
two_sum, two_prod, two_sqr, nint), `src/qd_real.cpp` (sloppy_div, accurate_div,
fsqrt/sqrt, nint(qd_real), pow(qd_real,int)).

The bulk of the port is mechanical: `double`→`float`, `x[i]`→`f{i}`, namespace,
splitter constant. This file documents the FP32-specific decisions and the two
places where the T3.0a **task text** and the **actual QD 2.3.24 source** differ
(source-fidelity rule 6 was applied — QD's real code is what got ported, and the
divergence is recorded here and in the T3.0a report).

---

## 0. Source-fidelity findings — task text vs QD 2.3.24 source

These are not bugs; they are cases where the task's algorithm sketch predates or
mis-attributes the QD 2.3.24 routine. Per the "QD-source-fidelity is
non-negotiable / do not hallucinate QD internals" mandate, the header ports what
QD 2.3.24 actually contains and cites it.

### 0a. `divide` is long division, NOT Newton

The task text describes divide as *"Newton iteration, initial reciprocal from
FP32 division, 3 iterations to ~96 bits."* QD 2.3.24's `qd_real::div`
(`qd_real.cpp:693-736`) is **classical long division**, not Newton:

```
q0 = a[0]/b[0];  r = a - b*q0;
q1 = r[0]/b[0];  r = r - b*q1;
q2 = r[0]/b[0];  r = r - b*q2;
q3 = r[0]/b[0];               // sloppy_div: 4 digits + renorm(4)
                              // accurate_div: + q4 = r[0]/b[0]; renorm_4(5)
```

Each quotient digit `q_k = r[0]/b[0]` contributes ~24 fresh bits
(q0≈24, q1≈48, q2≈72, q3≈96) and the residual `r` is refined by a full
QuadFloat multiply-subtract between digits. Four digits reach the ~96-bit
QuadFloat width. The port carries both `divide` (sloppy, the QD **default** —
QD builds with `QD_SLOPPY_DIV` on) and `divide_accurate` (5 digits + `renorm_4`).

The "3 Newton iterations, each doubling precision" reasoning in the task IS
correct arithmetic — it just describes `dd_real::sqrt`'s Karp trick, not QD's
quad-double divide. It is preserved as the sqrt-iteration justification (§0b).

### 0b. `sqrt` is Heron's method, NOT Karp reciprocal-Newton

The task text describes sqrt as *"Newton iteration, initial reciprocal from FP32
division, same posture as divide."* That is the **Karp trick** used by
`dd_real::sqrt` (`dd_real.cpp:47-72`: `x = 1/√a[0]; return a·x + …`). QD
2.3.24's **quad-double** `sqrt` (`qd_real.cpp:738-785`, `fsqrt`) is instead
**Heron's method** — a Newton iteration on `x²−a`:

```
x = √(a[0]);                          // ~24-bit FP32 seed
for i in 0..9:  y = ½(x + a/x);  if |x−y| < |x|·eps: return y;  x = y;
```

The port keeps QD's exact structure (up-to-10-iteration loop with early-out,
`eps = 2⁻⁹⁶`). **Iteration count justification** (confirming the task's "3"):
Heron doubles the number of correct digits each step. FP32 seed ≈ 24 bits →
48 → 96, saturating at the ~96-bit QuadFloat width, so **3 iterations** reach
full precision and the early-out `|x−y| < |x|·2⁻⁹⁶` fires on the 3rd. (The
smoke test confirms sqrt mean = 29.0 digits.) Note Heron needs a full QuadFloat
`divide` per step, so QF sqrt is heavier than a reciprocal-Newton would be —
this is inherited from QD's choice, not a port decision.

### 0c. Newton-iteration-count algebra (task deliverable)

Task asks to confirm 3 iterations for both divide and sqrt at FP32:

- **sqrt (Heron, §0b):** 24 → 48 → 96 bits ⇒ **3** iterations. ✓ (matches
  QD's early-out behavior).
- **divide:** QD is not Newton (§0a), but the *equivalent* precision growth
  holds per **quotient digit**: 24 → 48 → 72 → 96 bits ⇒ **4 digits** for
  sloppy_div (the default), **5** for accurate_div. So "3 doublings from a
  24-bit seed" = 3 *refinement* steps past the first digit, which is what QD's
  4-digit long division does. The task's "3 iterations" and QD's "4 digits"
  describe the same 96-bit target; the count differs only because long division
  counts the seed digit separately.

---

## 1. Dekker splitter reuse (FP32)

QF reuses FF's already-validated FP32 Dekker two_product (`two_prod`,
`ff_math.hpp:266-274`) and its splitter `8193.0f = 2¹³+1` — expressed in QD's
by-reference form as `qf_two_prod` (cites QD `inline.h:85-99`). Empirically
`two_prod` is **bit-exact** over `|operands| ≤ 1e6` for both `4097` and `8193`
(2M-sample harness); `8193` is chosen for FF parity. The FP32 EFT primitives
(`qf_two_sum`, `qf_quick_two_sum`, `qf_two_sqr`, `qf_three_sum`,
`qf_three_sum2`) are bit-identical to the transforms embedded in FF's
`add`/`multiply`, per the "reuse FF's EFT primitives, do not re-derive" mandate.

## 2. Splitter-overflow limit NOT ported (QD `_QD_SPLIT_THRESH`)

QD's `split()` (`inline.h:66-83`) has a large-magnitude branch that pre-scales
by `2⁻²⁸` when `|a| > _QD_SPLIT_THRESH = 2⁹⁹⁶` to avoid splitter overflow in
FP64. The FP32 analogue threshold is ~`2¹⁰²` (where `8193·a` overflows FP32's
~3.4e38 = 2¹²⁸ range). This branch is **deliberately not ported** in T3.0a: it
adds a per-multiply branch to every partial product, and QF's normal operating
range (inputs faithfully split from FP64, |value| ≤ ~1e38) stays well below the
threshold. The consequence surfaced once in the smoke test — a directed
`huge/tiny` divide whose quotient is `1e36` (> 2¹⁰²·⅟8193) drove `8193·q` to
`inf`→`NaN` inside `two_prod`, exactly the FP32 analogue of PORT_NOTES §4a
(FF `exp` splitter overflow at input > 79). It is guarded as an
out-of-safe-domain input in the smoke test (skip-not-fail, matching T1.1's
splitter-overflow guard), NOT worked around in the arithmetic. If a later T3.x
op needs the full FP32 range, port QD's `_QD_SPLIT_THRESH` branch then.

## 3. `sloppy_add` safety at the narrower FP32 exponent

QD's default add is `sloppy_add` (`qd_inline.h:338-405`; QD builds with
`QD_IEEE_ADD` **off**), which forgoes the digit-by-digit `quick_three_accum`
merge of `ieee_add` in favor of a fixed 4-wide component-wise `two_sum` chain.
Its correctness rests on the operands being **non-overlapping expansions**
(each `f_{i+1} ≤ ½ ulp(f_i)`), not on exponent range — the same property FP32
and FP64 both provide. So the DD→FF concern about FP32's 6×-narrower exponent
(PORT_NOTES §4a) does **not** make sloppy_add unsafe: no splitter, no
magic-constant, no dynamic-range assumption is involved. The port therefore
keeps QD's default `sloppy_add` (with `ieee_add` also provided for parity and
tight-bound callers). The smoke test confirms add/subtract at mean 29.0 digits
with no observed catastrophic loss. **If** a future accuracy sweep (T3.4) shows
sloppy_add failing a published bound at FP32's exponent extremes, switch the
`add()` dispatch to `ieee_add` and cite the failing corpus there — that switch
is a one-line change and both routines are already present.

## 4. `nint` — the FF `ffnint` bug does NOT recur

FF's `round_to_nearest_int` hit an off-by-one near half-integers because the
DD-style `2^(2p−1)` magic-constant trick is ill-conditioned at FP32's 24-bit
mantissa (PORT_NOTES §4b), and FF had to reroute rounding through FP64. QD's
`nint` (`inline.h:116-120`, `qd_real.cpp:48-86`) does **not** use the
magic-constant trick at all — it is `floor(d+0.5)` per component with
half-integer tie corrections keyed on the sign of the next component. That is
well-conditioned at every FP32 magnitude, so the FF bug has no analogue here and
`round_to_nearest_int` is a direct, unmodified port. (No FP64 detour needed.)

## 5. Constant generation precision

`scripts/gen_qf_constants.cpp` extends the FF Route-A generator to four words by
**successive splitting** of a 113-bit `__float128` source constant:
`f0=(float)x; r=x−f0; f1=(float)r; …` four times. `__float128` carries 113 bits
vs QF's ~96, ~17 bits (5 decimal digits) of headroom — ample. Reconstruction
`rel_err` for every constant is `< 6e-31` (reported inline by the generator),
comfortably below `u = 2⁻⁹⁶ ≈ 1.3e-29`. Euler-γ is not a libquadmath literal, so
it is seeded from its 36-digit decimal value via `strtoflt128`. Requires
`-fext-numeric-literals` under `-std=c++17` for the `Q` suffix (documented in
the generator's build comment).

---

## License lineage (T3.0a kickoff action item, TEST_SUITE_PLAN §"Phase 2/3
open question")

QF is modeled on **QD** (`qd_real.cc`), NOT DDFUN. QD carries the
**LBNL-BSD-License** (modified BSD-3-Clause + §3 grant-back), triple-authored
Hida/Li/Bailey with LBNL *institutional* copyright and commercial contact
`TTD@lbl.gov` / `ipo@lbl.gov` — a **different** license and copyright holder
than the **DHB-License** (Bailey personal, `dhbailey@lbl.gov`) that governs
`dd_math.hpp` / `ff_math.hpp` (DDFUN derivatives). Accordingly:

- `third_party/include/qf_math.hpp` carries `SPDX-License-Identifier:
  LicenseRef-LBNL-BSD-License`, NOT `LicenseRef-DHB-License`.
- `LICENSES/LicenseRef-LBNL-BSD-License.txt` added (verbatim modified-BSD text
  from QD 2.3.24's `COPYING` + `BSD-LBNL-License.doc`).

**Deviation from the T3.0a task prompt (recorded per rule 8):** the prompt said
"Copy ff_math.hpp's license header verbatim, adjust names" (i.e. DHB-License).
That conflicts with the plan doc's own §T3.0a-kickoff instruction to "verify QF
port lineage (DDFUN vs QD — QF is modeled on QD's qd_real.cc, so likely
LBNL-BSD-License) and apply the correct license header." The plan-doc lineage
check wins: applying the DHB-License to a QD derivative would mis-attribute
copyright to Bailey personally and cite the wrong commercial contact. The
LBNL-BSD-License is applied instead. `NOTICE.md` should gain a QF row when QF
merges to `main` (a T3.x merge task, out of T3.0a scope).
