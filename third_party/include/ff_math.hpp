// SPDX-License-Identifier: LicenseRef-DHB-License
//
// Copyright (c) 2024 David H. Bailey — DDFUN v04 (original algorithms)
// Modifications Copyright (c) 2026 UChicago Argonne, LLC
//
// This file is a mechanical translation of dd_math.hpp (this repo's
// Kokkos C++ port of DDFUN v04) from double-double (2×FP64) to
// float-float (2×FP32). Function inventory, algorithm choices, and
// coefficient tables descend from DDFUN v04 by David H. Bailey.
//
// FP32-specific modifications (input narrowing, splitter constant
// 8193.0f = 2^13+1, joint sin/cos doublings, Taylor branches for
// |a|<0.5 in sinh/cosh/atanh, direct exp scaling to avoid splitter
// overflow, nint magic-constant replacement) are documented in
// PORT_NOTES.md. These modifications fall under DHB-License §3
// (grant-back) and are governed by the same terms as the original.
//
// See LICENSES/LicenseRef-DHB-License.txt for the full license text
// and NOTICE.md for the per-file license mapping.

#pragma once

// Float-float real arithmetic — Kokkos::Experimental::FloatFloat.
// All functions KOKKOS_INLINE_FUNCTION (host + device via Kokkos/CUDA).
// Mechanically ported from dd_math.hpp (this repo's Kokkos C++ port of DDFUN
// v04 by David H. Bailey), swapping 2×FP64 for 2×FP32. See PORT_NOTES.md for
// the FP32-specific fixes.
//
// Precision: ~14.4 decimal digits (24-bit FP32 mantissa × 2 = 48 bits).
// Range: bounded by FP32 (~3.4e38), much tighter than FP64.
//
// Naming conventions (T0.4/T2.0, for eventual upstreaming to Kokkos) mirror
// dd_math.hpp:
//   * Type + math live under namespace Kokkos::Experimental so an upstream PR is
//     a mechanical move rather than a rewrite.
//   * Arithmetic free functions use STL-style names (add/subtract/multiply/
//     divide/negate) and are also reachable through operator overloads.
//   * Constants are free functions FloatFloat_pi(), FloatFloat_e(), ...
//   * The former bit-pattern constructor became the static factory
//     FloatFloat::from_bits(hi, lo).
//   * Every single/double-return math function is additionally re-exposed under
//     namespace Kokkos at the bottom of this header (forwarding overloads,
//     mirroring impl/Kokkos_QuadPrecisionMath.hpp) so Kokkos::exp(ff) works
//     identically to Kokkos::exp(double)/Kokkos::exp(__float128). Math functions
//     are also ADL-findable via the argument's namespace (Kokkos::Experimental).
//     add/subtract/multiply/divide are NOT re-exposed under Kokkos — they are for
//     operators + explicit ADL only.

#include <Kokkos_Core.hpp>
#include <cstdint>
#include <cstring>
#include <cmath>

#ifndef __CUDA_ARCH__
#  include <iomanip>
#  include <ostream>
#endif

namespace Kokkos {
namespace Experimental {

// ============================================================
// Forward declarations
// ============================================================
struct FloatFloat;
KOKKOS_INLINE_FUNCTION FloatFloat add(FloatFloat a, FloatFloat b);
KOKKOS_INLINE_FUNCTION FloatFloat subtract(FloatFloat a, FloatFloat b);
KOKKOS_INLINE_FUNCTION FloatFloat multiply(FloatFloat a, FloatFloat b);
KOKKOS_INLINE_FUNCTION FloatFloat divide(FloatFloat a, FloatFloat b);
KOKKOS_INLINE_FUNCTION FloatFloat multiply_scalar(FloatFloat a, float b);
KOKKOS_INLINE_FUNCTION FloatFloat divide_scalar(FloatFloat a, float b);
KOKKOS_INLINE_FUNCTION FloatFloat negate(FloatFloat a);
KOKKOS_INLINE_FUNCTION FloatFloat abs(FloatFloat a);
KOKKOS_INLINE_FUNCTION FloatFloat sqrt(FloatFloat a);
KOKKOS_INLINE_FUNCTION FloatFloat round_to_nearest_int(FloatFloat a);
KOKKOS_INLINE_FUNCTION FloatFloat pow_int(FloatFloat a, int n);
KOKKOS_INLINE_FUNCTION FloatFloat exp(FloatFloat a);
KOKKOS_INLINE_FUNCTION FloatFloat log(FloatFloat a);
KOKKOS_INLINE_FUNCTION FloatFloat pow(FloatFloat a, FloatFloat b);
KOKKOS_INLINE_FUNCTION void   sinhcosh(FloatFloat a, FloatFloat& x, FloatFloat& y);
KOKKOS_INLINE_FUNCTION void   sincos(FloatFloat a, FloatFloat& x, FloatFloat& y);
KOKKOS_INLINE_FUNCTION FloatFloat angle(FloatFloat x, FloatFloat y);

// ============================================================
// FloatFloat struct
// ============================================================
struct FloatFloat {
    float hi;
    float lo;

    KOKKOS_INLINE_FUNCTION FloatFloat() : hi(0.0f), lo(0.0f) {}
    KOKKOS_INLINE_FUNCTION FloatFloat(float h) : hi(h), lo(0.0f) {}
    KOKKOS_INLINE_FUNCTION FloatFloat(float h, float l) : hi(h), lo(l) {}
    KOKKOS_INLINE_FUNCTION FloatFloat(double h) : hi((float)h), lo((float)(h - (double)(float)h)) {}
    KOKKOS_INLINE_FUNCTION FloatFloat(const FloatFloat& o) : hi(o.hi), lo(o.lo) {}
    KOKKOS_INLINE_FUNCTION FloatFloat& operator=(const FloatFloat& o) { hi=o.hi; lo=o.lo; return *this; }

    // Factory: build a FloatFloat from the IEEE-754 bit patterns of its two
    // components. Safe on host (memcpy) and device (__int_as_float). Replaces
    // the former free bit-pattern constructor function make_ff().
    static KOKKOS_INLINE_FUNCTION FloatFloat from_bits(uint32_t hi_bits, uint32_t lo_bits) {
        float h, l;
#ifndef __CUDA_ARCH__
        std::memcpy(&h, &hi_bits, sizeof(float));
        std::memcpy(&l, &lo_bits, sizeof(float));
#else
        h = __int_as_float(static_cast<int>(hi_bits));
        l = __int_as_float(static_cast<int>(lo_bits));
#endif
        return FloatFloat(h, l);
    }

    KOKKOS_INLINE_FUNCTION FloatFloat operator-() const { return negate(*this); }
    KOKKOS_INLINE_FUNCTION FloatFloat operator+(FloatFloat b) const { return add(*this, b); }
    KOKKOS_INLINE_FUNCTION FloatFloat operator-(FloatFloat b) const { return subtract(*this, b); }
    KOKKOS_INLINE_FUNCTION FloatFloat operator*(FloatFloat b) const { return multiply(*this, b); }
    KOKKOS_INLINE_FUNCTION FloatFloat operator/(FloatFloat b) const { return divide(*this, b); }
    KOKKOS_INLINE_FUNCTION FloatFloat operator*(float b)  const { return multiply_scalar(*this, b); }
    KOKKOS_INLINE_FUNCTION FloatFloat operator/(float b)  const { return divide_scalar(*this, b); }
    KOKKOS_INLINE_FUNCTION FloatFloat operator+(float b)  const { return add(*this, FloatFloat(b)); }
    KOKKOS_INLINE_FUNCTION FloatFloat operator-(float b)  const { return subtract(*this, FloatFloat(b)); }

    KOKKOS_INLINE_FUNCTION FloatFloat& operator+=(FloatFloat b) { *this = *this + b; return *this; }
    KOKKOS_INLINE_FUNCTION FloatFloat& operator-=(FloatFloat b) { *this = *this - b; return *this; }
    KOKKOS_INLINE_FUNCTION FloatFloat& operator*=(FloatFloat b) { *this = *this * b; return *this; }
    KOKKOS_INLINE_FUNCTION FloatFloat& operator/=(FloatFloat b) { *this = *this / b; return *this; }
    KOKKOS_INLINE_FUNCTION FloatFloat& operator+=(float b) { *this = *this + b; return *this; }
    KOKKOS_INLINE_FUNCTION FloatFloat& operator-=(float b) { *this = *this - b; return *this; }
    KOKKOS_INLINE_FUNCTION FloatFloat& operator*=(float b) { *this = multiply_scalar(*this, b); return *this; }
    KOKKOS_INLINE_FUNCTION FloatFloat& operator/=(float b) { *this = divide_scalar(*this, b); return *this; }

    KOKKOS_INLINE_FUNCTION bool operator==(FloatFloat b) const { return hi==b.hi && lo==b.lo; }
    KOKKOS_INLINE_FUNCTION bool operator!=(FloatFloat b) const { return !(*this == b); }
    KOKKOS_INLINE_FUNCTION bool operator<(FloatFloat b)  const { return hi<b.hi || (hi==b.hi && lo<b.lo); }
    KOKKOS_INLINE_FUNCTION bool operator>(FloatFloat b)  const { return hi>b.hi || (hi==b.hi && lo>b.lo); }
    KOKKOS_INLINE_FUNCTION bool operator<=(FloatFloat b) const { return !(b < *this); }
    KOKKOS_INLINE_FUNCTION bool operator>=(FloatFloat b) const { return !(*this < b); }
};

KOKKOS_INLINE_FUNCTION FloatFloat operator+(float a, FloatFloat b) { return add(FloatFloat(a), b); }
KOKKOS_INLINE_FUNCTION FloatFloat operator-(float a, FloatFloat b) { return subtract(FloatFloat(a), b); }
KOKKOS_INLINE_FUNCTION FloatFloat operator*(float a, FloatFloat b) { return multiply_scalar(b, a); }
KOKKOS_INLINE_FUNCTION FloatFloat operator/(float a, FloatFloat b) { return divide(FloatFloat(a), b); }

#ifndef __CUDA_ARCH__
inline std::ostream& operator<<(std::ostream& os, const FloatFloat& d) {
    os << "[" << std::setprecision(8) << std::scientific << d.hi
       << ", " << d.lo << "]";
    return os;
}
#endif

// ============================================================
// Constants via bit-pattern construction (safe on host + device)
// ============================================================
// Auto-generated by scripts/gen_ff_constants.cpp -- do not edit by hand.
// Route A: round_to_nearest_FF(Bailey FP64 hi+lo pair).
KOKKOS_INLINE_FUNCTION FloatFloat FloatFloat_pi          () { return FloatFloat::from_bits(0x40490fdbU, 0xb3bbbd2eU); } // pi
KOKKOS_INLINE_FUNCTION FloatFloat FloatFloat_e           () { return FloatFloat::from_bits(0x402df854U, 0x33b14577U); } // e
KOKKOS_INLINE_FUNCTION FloatFloat FloatFloat_log2        () { return FloatFloat::from_bits(0x3f317218U, 0xb102e308U); } // ln(2)
KOKKOS_INLINE_FUNCTION FloatFloat FloatFloat_log10       () { return FloatFloat::from_bits(0x40135d8eU, 0xb309555dU); } // ln(10)
KOKKOS_INLINE_FUNCTION FloatFloat FloatFloat_sqrt2       () { return FloatFloat::from_bits(0x3fb504f3U, 0x32cfe77aU); } // sqrt(2)
KOKKOS_INLINE_FUNCTION FloatFloat FloatFloat_euler_gamma () { return FloatFloat::from_bits(0x3f13c468U, 0xb1e4127aU); } // Euler gamma

// ============================================================
// Primitive arithmetic
// ============================================================

KOKKOS_INLINE_FUNCTION FloatFloat negate(FloatFloat a) {
    return FloatFloat(-a.hi, -a.lo);
}

// TwoSum (Knuth)
KOKKOS_INLINE_FUNCTION FloatFloat add(FloatFloat a, FloatFloat b) {
    float t1 = a.hi + b.hi;
    float e  = t1 - a.hi;
    float t2 = ((b.hi - e) + (a.hi - (t1 - e))) + a.lo + b.lo;
    float hi = t1 + t2;
    float lo = t2 - (hi - t1);
    return FloatFloat(hi, lo);
}

KOKKOS_INLINE_FUNCTION FloatFloat subtract(FloatFloat a, FloatFloat b) {
    float t1 = a.hi - b.hi;
    float e  = t1 - a.hi;
    float t2 = ((-b.hi - e) + (a.hi - (t1 - e))) + a.lo - b.lo;
    float hi = t1 + t2;
    float lo = t2 - (hi - t1);
    return FloatFloat(hi, lo);
}

// TwoProduct (Dekker splitting). Splitter = 2^13 + 1 for FP32 (24-bit mantissa).
KOKKOS_INLINE_FUNCTION FloatFloat multiply(FloatFloat a, FloatFloat b) {
    const float split = 8193.0f;
    float cona = a.hi * split, conb = b.hi * split;
    float a1 = cona - (cona - a.hi), b1 = conb - (conb - b.hi);
    float a2 = a.hi - a1,            b2 = b.hi - b1;
    float c11 = a.hi * b.hi;
    float c21 = (((a1*b1 - c11) + a1*b2) + a2*b1) + a2*b2;
    float c2  = a.hi * b.lo + a.lo * b.hi;
    float t1  = c11 + c2;
    float e   = t1 - c11;
    float t2  = ((c2 - e) + (c11 - (t1 - e))) + c21 + a.lo * b.lo;
    float hi  = t1 + t2;
    float lo  = t2 - (hi - t1);
    return FloatFloat(hi, lo);
}

KOKKOS_INLINE_FUNCTION FloatFloat divide(FloatFloat a, FloatFloat b) {
    const float split = 8193.0f;
    // B8: the Dekker splitter below computes conb = b.hi * split (line ~"conb =")
    // to extract b.hi's high half. For |b.hi| > FLT_MAX / (split + 1) ≈ 4.15e34
    // that product overflows to ±inf, then b1 = conb - (conb - b.hi) = inf - inf
    // = NaN, which poisons the whole quotient. This bites log()'s Newton
    // iteration (divisor exp(b) ≈ a) for x ≳ e^79.7 ≈ 1.5e34, feeding NaN back
    // into exp() and hanging its Taylor loop at the 60-iteration cap (surfaced by
    // the B4 investigation — see TEST_SUITE_PLAN.md §B4/§B8).
    //
    // Fix: when b.hi is in the overflow-hazard band, pre-scale the divisor down
    // by an exact power of two (2^-64: no FP rounding, so b's full FF precision is
    // preserved), run the unchanged Dekker split, then unscale the quotient. Since
    // q = a / (b·s) = (a/b) / s, the true quotient a/b is recovered by MULTIPLYING
    // q by the same down-scale factor s. 2^-64 gives ample headroom: the largest
    // |b.hi| ≈ FLT_MAX = 2^128 maps to 2^64, and 2^64 * split ≈ 2^77 ≪ FLT_MAX.
    // Mirrors PORT_NOTES §4a's power-of-2 scaling pattern for exp's final scaling
    // (same bug class, different site — the splitter, not exp's scaling).
    const float kSplitOverflowThresh = 4.1528233e34f; // FLT_MAX / (split + 1)
    const float s = (Kokkos::fabs(b.hi) > kSplitOverflowThresh)
                        ? ldexpf(1.0f, -64) : 1.0f;
    b = FloatFloat(b.hi * s, b.lo * s);
    float s1   = a.hi / b.hi;
    float cona = s1 * split, conb = b.hi * split;
    float a1   = cona - (cona - s1), b1 = conb - (conb - b.hi);
    float a2   = s1 - a1,            b2 = b.hi - b1;
    float c11  = s1 * b.hi;
    float c21  = (((a1*b1 - c11) + a1*b2) + a2*b1) + a2*b2;
    float c2   = s1 * b.lo;
    float t1   = c11 + c2;
    float e    = t1 - c11;
    float t2   = ((c2 - e) + (c11 - (t1 - e))) + c21;
    float t12  = t1 + t2;
    float t22  = t2 - (t12 - t1);
    float t11  = a.hi - t12;
    e = t11 - a.hi;
    float t21  = ((-t12 - e) + (a.hi - (t11 - e))) + a.lo - t22;
    float s2   = (t11 + t21) / b.hi;
    float hi   = s1 + s2;
    float lo   = s2 - (hi - s1);
    // B8: unscale — recover a/b from a/(b·s) by multiplying by s (exact power of 2
    // when the pre-scale fired; s == 1.0f is a no-op on the non-overflow path).
    return FloatFloat(hi * s, lo * s);
}

KOKKOS_INLINE_FUNCTION FloatFloat multiply_scalar(FloatFloat a, float b) {
    const float split = 8193.0f;
    float cona = a.hi * split, conb = b * split;
    float a1   = cona - (cona - a.hi), b1 = conb - (conb - b);
    float a2   = a.hi - a1,            b2 = b - b1;
    float c11  = a.hi * b;
    float c21  = (((a1*b1 - c11) + a1*b2) + a2*b1) + a2*b2;
    float c2   = a.lo * b;
    float t1   = c11 + c2;
    float e    = t1 - c11;
    float t2   = ((c2 - e) + (c11 - (t1 - e))) + c21;
    float hi   = t1 + t2;
    float lo   = t2 - (hi - t1);
    return FloatFloat(hi, lo);
}

KOKKOS_INLINE_FUNCTION FloatFloat divide_scalar(FloatFloat a, float b) {
    const float split = 8193.0f;
    float t1   = a.hi / b;
    float cona = t1 * split, conb = b * split;
    float a1   = cona - (cona - t1), b1 = conb - (conb - b);
    float a2   = t1 - a1,            b2 = b - b1;
    float t12  = t1 * b;
    float t22  = (((a1*b1 - t12) + a1*b2) + a2*b1) + a2*b2;
    float t11  = a.hi - t12;
    float e    = t11 - a.hi;
    float t21  = ((-t12 - e) + (a.hi - (t11 - e))) + a.lo - t22;
    float t2   = (t11 + t21) / b;
    float hi   = t1 + t2;
    float lo   = t2 - (hi - t1);
    return FloatFloat(hi, lo);
}

// Exact product of two floats
KOKKOS_INLINE_FUNCTION FloatFloat two_prod(float fa, float fb) {
    const float split = 8193.0f;
    float cona = fa * split, conb = fb * split;
    float a1   = cona - (cona - fa), b1 = conb - (conb - fb);
    float a2   = fa - a1,            b2 = fb - b1;
    float s1   = fa * fb;
    float s2   = (((a1*b1 - s1) + a1*b2) + a2*b1) + a2*b2;
    return FloatFloat(s1, s2);
}

// ============================================================
// Basic math
// ============================================================

KOKKOS_INLINE_FUNCTION FloatFloat abs(FloatFloat a) {
    return (a.hi >= 0.0f) ? a : FloatFloat(-a.hi, -a.lo);
}

// Nearest integer. The DD-style magic-constant trick (using a 2^47 FF constant)
// is fragile in FP32: ULP at 2^47 is 2^24, much larger than typical integer
// inputs, so the FF lo component must rescue the precision and ties land on
// the wrong side. Instead, do the rounding in FP64 — FF values are bounded by
// 2^48 and fit exactly in FP64's 53-bit mantissa, where the magic-constant
// trick is well-conditioned.
KOKKOS_INLINE_FUNCTION FloatFloat round_to_nearest_int(FloatFloat a) {
    if (a.hi == 0.0f) return FloatFloat(0.0f);
    double total = (double)a.hi + (double)a.lo;
    if (Kokkos::fabs(total) >= 1.40737488355328e14 /* 2^47 */) {
        Kokkos::printf("FFNINT: argument too large\n");
        return FloatFloat(0.0f);
    }
    const double T52 = 4.503599627370496e15; // 2^52
    double rounded = (total > 0.0) ? (total + T52) - T52 : (total - T52) + T52;
    float hi = (float)rounded;
    float lo = (float)(rounded - (double)hi);
    return FloatFloat(hi, lo);
}

KOKKOS_INLINE_FUNCTION FloatFloat sqrt(FloatFloat a) {
    if (a.hi == 0.0f) return FloatFloat(0.0f);
    if (a.hi < 0.0f) {
        Kokkos::printf("FFSQRT: negative argument\n");
        return FloatFloat(0.0f);
    }
    float t1 = 1.0f / Kokkos::sqrt(a.hi);
    float t2 = a.hi * t1;
    FloatFloat s0 = two_prod(t2, t2);
    FloatFloat s1 = subtract(a, s0);
    float t3  = 0.5f * s1.hi * t1;
    return add(FloatFloat(t2), FloatFloat(t3));
}

// Integer power
KOKKOS_INLINE_FUNCTION FloatFloat pow_int(FloatFloat a, int n) {
    const float cl2 = 1.4426950408889633f;
    if (a.hi == 0.0f) {
        if (n >= 0) return FloatFloat(0.0f);
        Kokkos::printf("FFNPWR: zero base with negative exponent\n");
        return FloatFloat(0.0f);
    }
    int nn = (n < 0) ? -n : n;
    if (nn == 0) return FloatFloat(1.0f);
    if (nn == 1) return (n > 0) ? a : divide(FloatFloat(1.0f), a);
    if (nn == 2) { FloatFloat r = multiply(a,a); return (n>0) ? r : divide(FloatFloat(1.0f),r); }
    int mn = (int)(cl2 * Kokkos::log((float)nn) + 1.0f + 1.0e-6f);
    FloatFloat s0 = a, s2 = FloatFloat(1.0f);
    int kn = nn;
    for (int j = 1; j <= mn; ++j) {
        int kk = kn / 2;
        if (kn != 2*kk) s2 = multiply(s2, s0);
        kn = kk;
        if (j < mn) s0 = multiply(s0, s0);
    }
    if (n < 0) s2 = divide(FloatFloat(1.0f), s2);
    return s2;
}

// ============================================================
// Exp / Log family
// ============================================================

KOKKOS_INLINE_FUNCTION FloatFloat exp(FloatFloat a) {
    const int nq = 4;
    const float eps = 1.0e-15f;
    FloatFloat al2 = FloatFloat_log2();
    // FP32 finite range: |x| < ln(3.4e38) ~= 88.7
    if (a.hi >= 88.0f) {
        Kokkos::printf("FFEXP: argument too large\n");
        return FloatFloat(0.0f);
    }
    if (a.hi <= -88.0f) return FloatFloat(0.0f);

    FloatFloat s0 = divide(a, al2);
    FloatFloat s1 = round_to_nearest_int(s0);
    float t1  = s1.hi;
    int nz    = (int)(t1 + Kokkos::copysign(1.0e-6f, t1));
    s0 = subtract(a, multiply(al2, s1));

    if (s0.hi == 0.0f) {
        return FloatFloat(ldexpf(1.0f, nz));
    }
    // Scale down by 2^nq then square nq times
    s1 = multiply_scalar(s0, ldexpf(1.0f, -nq));
    FloatFloat s2 = FloatFloat(1.0f), s3 = FloatFloat(1.0f);
    for (int l1 = 1; l1 <= 60; ++l1) {
        s0 = multiply(s2, s1);
        s2 = divide_scalar(s0, (float)l1);
        s0 = add(s3, s2);
        s3 = s0;
        if (Kokkos::fabs(s2.hi) <= eps * Kokkos::fabs(s3.hi)) break;
        if (l1 == 60) { Kokkos::printf("FFEXP: iteration limit\n"); return FloatFloat(0.0f); }
    }
    for (int i = 0; i < nq; ++i) s3 = multiply(s3, s3);

    // Final scaling by 2^nz is exact in FP32 (power-of-2 multiplication does
    // not round). Going through multiply_scalar would compute b*8193 inside Dekker
    // splitting, which overflows for nz >= 115 (i.e. a > ~79) — the cause of
    // the previous NaN outputs at the high end of the input range.
    float pow2 = ldexpf(1.0f, nz);
    return FloatFloat(s3.hi * pow2, s3.lo * pow2);
}

KOKKOS_INLINE_FUNCTION FloatFloat log(FloatFloat a) {
    if (a.hi <= 0.0f) {
        Kokkos::printf("FFLOG: non-positive argument\n");
        return FloatFloat(0.0f);
    }
    // Initial approximation then 2 Newton steps (FP32 base gives ~6 digits, doubles per iter -> 24 -> 48 bits)
    FloatFloat b = FloatFloat(Kokkos::log(a.hi));
    for (int k = 0; k < 2; ++k) {
        FloatFloat s0 = exp(b);
        FloatFloat s1 = subtract(a, s0);
        FloatFloat s2 = divide(s1, s0);
        b = add(b, s2);
    }
    return b;
}

KOKKOS_INLINE_FUNCTION FloatFloat log2(FloatFloat a) {
    return divide(log(a), FloatFloat_log2());
}

KOKKOS_INLINE_FUNCTION FloatFloat log10(FloatFloat a) {
    return divide(log(a), FloatFloat_log10());
}

KOKKOS_INLINE_FUNCTION FloatFloat log1p(FloatFloat a) {
    return log(add(FloatFloat(1.0f), a));
}

KOKKOS_INLINE_FUNCTION FloatFloat exp2(FloatFloat a) {
    return exp(multiply(a, FloatFloat_log2()));
}

KOKKOS_INLINE_FUNCTION FloatFloat exp10(FloatFloat a) {
    return exp(multiply(a, FloatFloat_log10()));
}

KOKKOS_INLINE_FUNCTION FloatFloat expm1(FloatFloat a) {
    if (Kokkos::fabs(a.hi) > 0.5f) {
        return subtract(exp(a), FloatFloat(1.0f));
    }
    // Taylor: a + a^2/2! + a^3/3! + ...
    FloatFloat sum = a, term = a;
    for (int k = 2; k <= 30; ++k) {
        term = divide_scalar(multiply(term, a), (float)k);
        sum  = add(sum, term);
        if (Kokkos::fabs(term.hi) < 1.0e-15f * Kokkos::fabs(sum.hi)) break;
    }
    return sum;
}

// ============================================================
// Trig — internal combined cos+sin, then derived
// ============================================================

// Track sin and cos jointly through nq doublings — avoids the sqrt(1-cos^2)
// recovery step, which loses relative precision when sin is near zero
// (i.e. when the answer most needs precision).
KOKKOS_INLINE_FUNCTION void sincos(FloatFloat a, FloatFloat& x, FloatFloat& y) {
    const int itrmx = 100, nq = 4;
    const float eps = 1.0e-15f;
    if (a.hi == 0.0f) { x = FloatFloat(1.0f); y = FloatFloat(0.0f); return; }
    if (a.hi >= 1.0e30f) {
        Kokkos::printf("FFCSSNR: argument too large\n");
        x = FloatFloat(0.0f); y = FloatFloat(0.0f); return;
    }
    FloatFloat pi2 = multiply_scalar(FloatFloat_pi(), 2.0f);
    FloatFloat s1  = divide(a, pi2);
    FloatFloat s2  = round_to_nearest_int(s1);
    FloatFloat s3  = subtract(a, multiply(pi2, s2));
    if (s3.hi == 0.0f) { x = FloatFloat(1.0f); y = FloatFloat(0.0f); return; }
    float scale = 1.0f / (float)(1 << nq);
    FloatFloat r  = multiply_scalar(s3, scale);   // r = s3 / 2^nq, |r| < pi/2^nq
    FloatFloat r2 = multiply(r, r);

    // sin(r) = r - r^3/3! + r^5/5! - ...
    // cos(r) = 1 - r^2/2! + r^4/4! - ...
    FloatFloat sin_r = r,             cos_r  = FloatFloat(1.0f);
    FloatFloat sterm = r,             cterm  = FloatFloat(1.0f);
    for (int k = 1; k <= itrmx; ++k) {
        sterm = divide_scalar(multiply(sterm, r2), -(float)((2*k) * (2*k + 1)));
        sin_r = add(sin_r, sterm);
        cterm = divide_scalar(multiply(cterm, r2), -(float)((2*k - 1) * (2*k)));
        cos_r = add(cos_r, cterm);
        if (Kokkos::fabs(sterm.hi) < eps * Kokkos::fabs(sin_r.hi) &&
            Kokkos::fabs(cterm.hi) < eps) break;
        if (k == itrmx) { Kokkos::printf("FFCSSNR: iteration limit\n"); return; }
    }

    // Doubling: sin(2x) = 2 sin x cos x, cos(2x) = cos^2 x - sin^2 x
    for (int j = 0; j < nq; ++j) {
        FloatFloat new_sin = multiply_scalar(multiply(sin_r, cos_r), 2.0f);
        FloatFloat new_cos = subtract(multiply(cos_r, cos_r), multiply(sin_r, sin_r));
        sin_r = new_sin;
        cos_r = new_cos;
    }

    x = cos_r; y = sin_r;
}

KOKKOS_INLINE_FUNCTION FloatFloat sin(FloatFloat a) {
    FloatFloat c, s; sincos(a, c, s); return s;
}
KOKKOS_INLINE_FUNCTION FloatFloat cos(FloatFloat a) {
    FloatFloat c, s; sincos(a, c, s); return c;
}
KOKKOS_INLINE_FUNCTION FloatFloat tan(FloatFloat a) {
    FloatFloat c, s; sincos(a, c, s); return divide(s, c);
}

// Angle of point (x, y) = atan2(y, x)
KOKKOS_INLINE_FUNCTION FloatFloat angle(FloatFloat x, FloatFloat y) {
    FloatFloat pi = FloatFloat_pi();
    if (x.hi == 0.0f && y.hi == 0.0f) return FloatFloat(0.0f);
    if (x.hi == 0.0f) return (y.hi > 0.0f) ? multiply_scalar(pi, 0.5f) : multiply_scalar(pi, -0.5f);
    if (y.hi == 0.0f) return (x.hi > 0.0f) ? FloatFloat(0.0f) : pi;
    FloatFloat r = sqrt(add(multiply(x,x), multiply(y,y)));
    FloatFloat nx = divide(x, r), ny = divide(y, r);
    FloatFloat a = FloatFloat(Kokkos::atan2(ny.hi, nx.hi));
    bool use_x = (Kokkos::fabs(nx.hi) <= Kokkos::fabs(ny.hi));
    FloatFloat target = use_x ? nx : ny;
    for (int k = 0; k < 3; ++k) {
        FloatFloat sin_a, cos_a;
        sincos(a, cos_a, sin_a);
        FloatFloat corr;
        if (use_x) {
            corr = divide(subtract(target, cos_a), sin_a);
            a = subtract(a, corr);
        } else {
            corr = divide(subtract(target, sin_a), cos_a);
            a = add(a, corr);
        }
    }
    return a;
}

KOKKOS_INLINE_FUNCTION FloatFloat asin(FloatFloat a) {
    if (Kokkos::fabs(a.hi) > 1.0f) {
        Kokkos::printf("FFASIN: argument out of range\n");
        return FloatFloat(0.0f);
    }
    FloatFloat t = sqrt(subtract(FloatFloat(1.0f), multiply(a, a)));
    return angle(t, a);
}
KOKKOS_INLINE_FUNCTION FloatFloat acos(FloatFloat a) {
    if (Kokkos::fabs(a.hi) > 1.0f) {
        Kokkos::printf("FFACOS: argument out of range\n");
        return FloatFloat(0.0f);
    }
    FloatFloat t = sqrt(subtract(FloatFloat(1.0f), multiply(a, a)));
    return angle(a, t);
}
KOKKOS_INLINE_FUNCTION FloatFloat atan(FloatFloat a) {
    return angle(FloatFloat(1.0f), a);
}
KOKKOS_INLINE_FUNCTION FloatFloat atan2(FloatFloat y, FloatFloat x) {
    return angle(x, y);
}

// ============================================================
// Hyperbolic
// ============================================================

KOKKOS_INLINE_FUNCTION void sinhcosh(FloatFloat a, FloatFloat& x, FloatFloat& y) {
    // Taylor series for |a| < 0.5 — avoids the (e^a - e^-a)/2 cancellation
    // when a is small (both exponentials approach 1, leading bits cancel).
    if (Kokkos::fabs(a.hi) < 0.5f) {
        FloatFloat a2 = multiply(a, a);
        FloatFloat sinh_sum = a,             sinh_term = a;
        FloatFloat cosh_sum = FloatFloat(1.0f),  cosh_term = FloatFloat(1.0f);
        for (int k = 1; k <= 30; ++k) {
            sinh_term = divide_scalar(multiply(sinh_term, a2), (float)((2*k) * (2*k + 1)));
            sinh_sum  = add(sinh_sum, sinh_term);
            cosh_term = divide_scalar(multiply(cosh_term, a2), (float)((2*k - 1) * (2*k)));
            cosh_sum  = add(cosh_sum, cosh_term);
            if (Kokkos::fabs(sinh_term.hi) < 1.0e-15f * Kokkos::fabs(sinh_sum.hi) &&
                Kokkos::fabs(cosh_term.hi) < 1.0e-15f) break;
        }
        x = cosh_sum; y = sinh_sum;
        return;
    }
    FloatFloat s0 = exp(a);
    FloatFloat s1 = divide(FloatFloat(1.0f), s0);
    x = multiply_scalar(add(s0, s1), 0.5f);
    y = multiply_scalar(subtract(s0, s1), 0.5f);
}

KOKKOS_INLINE_FUNCTION FloatFloat sinh(FloatFloat a) {
    FloatFloat c, s; sinhcosh(a, c, s); return s;
}
KOKKOS_INLINE_FUNCTION FloatFloat cosh(FloatFloat a) {
    FloatFloat c, s; sinhcosh(a, c, s); return c;
}
KOKKOS_INLINE_FUNCTION FloatFloat tanh(FloatFloat a) {
    if (a.hi < 0.0f) return negate(tanh(negate(a)));
    FloatFloat e = expm1(multiply_scalar(a, 2.0f));
    return divide(e, add(e, FloatFloat(2.0f)));
}

KOKKOS_INLINE_FUNCTION FloatFloat asinh(FloatFloat a) {
    if (a.hi < 0.0f) return negate(asinh(negate(a)));
    return log(add(a, sqrt(add(multiply(a, a), FloatFloat(1.0f)))));
}
KOKKOS_INLINE_FUNCTION FloatFloat acosh(FloatFloat a) {
    if (a.hi < 1.0f) { Kokkos::printf("FFACOSH: argument < 1\n"); return FloatFloat(0.0f); }
    FloatFloat t1 = subtract(multiply(a, a), FloatFloat(1.0f));
    return log(add(a, sqrt(t1)));
}
KOKKOS_INLINE_FUNCTION FloatFloat atanh(FloatFloat a) {
    if (Kokkos::fabs(a.hi) >= 1.0f) { Kokkos::printf("FFATANH: |argument| >= 1\n"); return FloatFloat(0.0f); }
    // Taylor for |a|<0.5 avoids calling log (which loses precision when its
    // argument is close to 1). All terms positive — no cancellation.
    if (Kokkos::fabs(a.hi) < 0.5f) {
        FloatFloat a2 = multiply(a, a);
        FloatFloat sum = a, pwr = a;
        for (int k = 1; k <= 60; ++k) {
            pwr  = multiply(pwr, a2);
            FloatFloat term = divide_scalar(pwr, (float)(2*k + 1));
            sum  = add(sum, term);
            if (Kokkos::fabs(term.hi) < 1.0e-15f * Kokkos::fabs(sum.hi)) break;
        }
        return sum;
    }
    // For 0.5 <= |a| < 1, log((1+a)/(1-a)) is well-conditioned (ratio >= 3).
    FloatFloat t1 = add(FloatFloat(1.0f), a);
    FloatFloat t2 = subtract(FloatFloat(1.0f), a);
    return multiply_scalar(log(divide(t1, t2)), 0.5f);
}

// ============================================================
// Multi-argument operations
// ============================================================

KOKKOS_INLINE_FUNCTION FloatFloat pow(FloatFloat a, FloatFloat b) {
    if (a.hi <= 0.0f) {
        if (a.hi == 0.0f && b.hi > 0.0f) return FloatFloat(0.0f);
        Kokkos::printf("FFPOW: non-positive base\n");
        return FloatFloat(0.0f);
    }
    return exp(multiply(log(a), b));
}

KOKKOS_INLINE_FUNCTION FloatFloat hypot(FloatFloat a, FloatFloat b) {
    return sqrt(add(multiply(a, a), multiply(b, b)));
}

KOKKOS_INLINE_FUNCTION FloatFloat ceil(FloatFloat a);
KOKKOS_INLINE_FUNCTION FloatFloat floor(FloatFloat a);
KOKKOS_INLINE_FUNCTION FloatFloat trunc(FloatFloat a);
KOKKOS_INLINE_FUNCTION FloatFloat round(FloatFloat a);

KOKKOS_INLINE_FUNCTION FloatFloat fmod(FloatFloat a, FloatFloat b) {
    FloatFloat q = divide(a, b);
    FloatFloat qt = trunc(q);
    return subtract(a, multiply(b, qt));
}

KOKKOS_INLINE_FUNCTION FloatFloat remainder(FloatFloat a, FloatFloat b) {
    FloatFloat q = divide(a, b);
    FloatFloat qn = round_to_nearest_int(q);
    return subtract(a, multiply(b, qn));
}

KOKKOS_INLINE_FUNCTION FloatFloat copysign(FloatFloat a, FloatFloat b) {
    FloatFloat r = abs(a);
    if (b.hi < 0.0f || (b.hi == 0.0f && b.lo < 0.0f)) return negate(r);
    return r;
}

KOKKOS_INLINE_FUNCTION FloatFloat fmax(FloatFloat a, FloatFloat b) {
    return (a > b) ? a : b;
}
KOKKOS_INLINE_FUNCTION FloatFloat fmin(FloatFloat a, FloatFloat b) {
    return (a < b) ? a : b;
}
KOKKOS_INLINE_FUNCTION FloatFloat fdim(FloatFloat a, FloatFloat b) {
    return (a > b) ? subtract(a, b) : FloatFloat(0.0f);
}
KOKKOS_INLINE_FUNCTION FloatFloat fma(FloatFloat a, FloatFloat b, FloatFloat c) {
    return add(multiply(a, b), c);
}

// ============================================================
// Rounding
// ============================================================

KOKKOS_INLINE_FUNCTION FloatFloat floor(FloatFloat a) {
    FloatFloat n = round_to_nearest_int(a);
    if (n > a) return subtract(n, FloatFloat(1.0f));
    return n;
}
KOKKOS_INLINE_FUNCTION FloatFloat ceil(FloatFloat a) {
    FloatFloat n = round_to_nearest_int(a);
    if (n < a) return add(n, FloatFloat(1.0f));
    return n;
}
KOKKOS_INLINE_FUNCTION FloatFloat trunc(FloatFloat a) {
    return (a.hi >= 0.0f) ? floor(a) : ceil(a);
}
KOKKOS_INLINE_FUNCTION FloatFloat round(FloatFloat a) {
    return round_to_nearest_int(a);
}

// ============================================================
// Special functions (in header, not benchmarked)
// ============================================================

KOKKOS_INLINE_FUNCTION FloatFloat erf(FloatFloat z) {
    const float eps = 1.0e-15f;
    if (z.hi == 0.0f) return FloatFloat(0.0f);
    const float large = 6.0f; // erfc(6) ~= 2e-17 << FF resolution
    if (z.hi >  large) return FloatFloat( 1.0f);
    if (z.hi < -large) return FloatFloat(-1.0f);

    FloatFloat z2 = multiply(z, z);
    int sign = (z.hi >= 0.0f) ? 1 : -1;
    FloatFloat az = abs(z);

    if (Kokkos::fabs(z.hi) < 4.0f) {
        FloatFloat t1 = FloatFloat(0.0f), t2 = az, t3 = FloatFloat(1.0f);
        for (int k = 0; k <= 60; ++k) {
            if (k > 0) {
                t2 = multiply_scalar(multiply(z2, t2), 2.0f);
                t3 = multiply_scalar(t3, 2.0f*k + 1.0f);
            }
            FloatFloat t4 = divide(t2, t3);
            FloatFloat t1new = add(t1, t4);
            if (Kokkos::fabs(t4.hi) < eps * Kokkos::fabs(t1new.hi)) { t1 = t1new; break; }
            t1 = t1new;
        }
        FloatFloat result = multiply_scalar(divide(multiply_scalar(t1, 2.0f),
                                multiply(sqrt(FloatFloat_pi()), exp(z2))), 1.0f);
        return (sign > 0) ? result : negate(result);
    } else {
        FloatFloat t1 = FloatFloat(0.0f), t2 = FloatFloat(1.0f), t3 = az;
        for (int k = 0; k <= 60; ++k) {
            if (k > 0) {
                t2 = multiply_scalar(t2, -(2.0f*k - 1.0f));
                t3 = multiply(t3, multiply_scalar(z2, 2.0f));
            }
            FloatFloat t4 = divide(t2, t3);
            FloatFloat t1new = add(t1, t4);
            if (Kokkos::fabs(divide(t4, t1new).hi) < eps) { t1 = t1new; break; }
            t1 = t1new;
        }
        FloatFloat erfc_val = divide(t1, multiply(sqrt(FloatFloat_pi()), exp(z2)));
        FloatFloat erf_val  = subtract(FloatFloat(1.0f), erfc_val);
        return (sign > 0) ? erf_val : negate(erf_val);
    }
}

KOKKOS_INLINE_FUNCTION FloatFloat erfc(FloatFloat z) {
    return subtract(FloatFloat(1.0f), erf(z));
}

// gamma — Lanczos approximation
KOKKOS_INLINE_FUNCTION FloatFloat tgamma(FloatFloat a) {
    if (a.hi < 0.5f) {
        FloatFloat pi = FloatFloat_pi();
        FloatFloat sin_pi_a = sin(multiply(pi, a));
        return divide(pi, multiply(sin_pi_a, tgamma(subtract(FloatFloat(1.0f), a))));
    }
    // B7: Lanczos g=7 coefficients promoted from `float` to `double`. Stored as
    // `float` literals, each coefficient was truncated to FP32's ~7-digit ceiling,
    // capping tgamma at ~6 digits regardless of the enclosing FF arithmetic (a
    // mechanical DD->FF port artifact: the `double` literals of dd_math.hpp:730-738
    // were erroneously given `f` suffixes). As `double`, each FloatFloat(cN) call
    // below invokes the FloatFloat(double) constructor (ff_math.hpp:94), which splits
    // to a full FF pair (hi=(float)d, lo=(float)(d-(double)hi)) — ~14 digits, the FF
    // resolution floor. double (53-bit) exceeds FF (48-bit), so the split is
    // FF-exact. Coefficient set: Godfrey g=7 (P. Godfrey 2001; same values as
    // dd_math.hpp's DD tgamma / Boost.Math / Wikipedia "Lanczos approximation").
    const double c0 =  0.99999999999980993;
    const double c1 =  676.5203681218851;
    const double c2 = -1259.1392167224028;
    const double c3 =  771.32342877765313;
    const double c4 = -176.61502916214059;
    const double c5 =  12.507343278686905;
    const double c6 = -0.13857109526572012;
    const double c7 =  9.9843695780195716e-6;
    const double c8 =  1.5056327351493116e-7;
    FloatFloat x = subtract(a, FloatFloat(1.0f));
    FloatFloat t = add(x, FloatFloat(7.5f));
    FloatFloat s = FloatFloat(c0);
    s = add(s, divide(FloatFloat(c1), add(x, FloatFloat(1.0f))));
    s = add(s, divide(FloatFloat(c2), add(x, FloatFloat(2.0f))));
    s = add(s, divide(FloatFloat(c3), add(x, FloatFloat(3.0f))));
    s = add(s, divide(FloatFloat(c4), add(x, FloatFloat(4.0f))));
    s = add(s, divide(FloatFloat(c5), add(x, FloatFloat(5.0f))));
    s = add(s, divide(FloatFloat(c6), add(x, FloatFloat(6.0f))));
    s = add(s, divide(FloatFloat(c7), add(x, FloatFloat(7.0f))));
    s = add(s, divide(FloatFloat(c8), add(x, FloatFloat(8.0f))));
    // B7: sqrt(2*pi) leading factor — likewise `double` (was `2.5...f`), so it
    // splits to a full FF pair instead of capping the whole product at ~7 digits.
    FloatFloat two_pi_sqrt = FloatFloat(2.5066282746310002);
    return multiply(multiply(two_pi_sqrt, s),
                 multiply(pow(t, add(x, FloatFloat(0.5f))), exp(negate(t))));
}

// Bessel J0 via series
KOKKOS_INLINE_FUNCTION FloatFloat bessel_j0(FloatFloat x) {
    const float eps = 1.0e-15f;
    FloatFloat x2 = multiply_scalar(multiply(x, x), -0.25f);
    FloatFloat term = FloatFloat(1.0f), sum = FloatFloat(1.0f);
    for (int k = 1; k <= 60; ++k) {
        term = divide_scalar(multiply(term, x2), (float)(k*k));
        sum  = add(sum, term);
        if (Kokkos::fabs(term.hi) < eps * Kokkos::fabs(sum.hi)) break;
    }
    return sum;
}

KOKKOS_INLINE_FUNCTION FloatFloat bessel_j1(FloatFloat x) {
    const float eps = 1.0e-15f;
    FloatFloat x2 = multiply_scalar(multiply(x, x), -0.25f);
    FloatFloat term = multiply_scalar(x, 0.5f), sum = term;
    for (int k = 1; k <= 60; ++k) {
        term = divide_scalar(multiply(term, x2), (float)(k * (k+1)));
        sum  = add(sum, term);
        if (Kokkos::fabs(term.hi) < eps * Kokkos::fabs(sum.hi)) break;
    }
    return sum;
}

KOKKOS_INLINE_FUNCTION FloatFloat bessel_jn(int n, FloatFloat x) {
    if (n == 0) return bessel_j0(x);
    if (n == 1) return bessel_j1(x);
    FloatFloat j0 = bessel_j0(x), j1 = bessel_j1(x);
    FloatFloat jm1 = j0, j_cur = j1;
    for (int k = 1; k < n; ++k) {
        FloatFloat jp1 = subtract(multiply_scalar(divide(j_cur, x), 2.0f*k), jm1);
        jm1   = j_cur;
        j_cur = jp1;
    }
    return j_cur;
}

KOKKOS_INLINE_FUNCTION FloatFloat bessel_y0(FloatFloat x) {
    FloatFloat two_over_pi = divide_scalar(FloatFloat(2.0f), FloatFloat_pi().hi);
    FloatFloat j0 = bessel_j0(x);
    return multiply(two_over_pi, multiply(j0, log(multiply_scalar(x, 0.5f))));
}
KOKKOS_INLINE_FUNCTION FloatFloat bessel_y1(FloatFloat x) {
    FloatFloat two_over_pi = divide_scalar(FloatFloat(2.0f), FloatFloat_pi().hi);
    FloatFloat j1 = bessel_j1(x);
    return multiply(two_over_pi, multiply(j1, log(multiply_scalar(x, 0.5f))));
}
KOKKOS_INLINE_FUNCTION FloatFloat bessel_yn(int n, FloatFloat x) {
    if (n == 0) return bessel_y0(x);
    if (n == 1) return bessel_y1(x);
    FloatFloat y0 = bessel_y0(x), y1 = bessel_y1(x);
    FloatFloat ym1 = y0, y_cur = y1;
    for (int k = 1; k < n; ++k) {
        FloatFloat yp1 = subtract(multiply_scalar(divide(y_cur, x), 2.0f*k), ym1);
        ym1   = y_cur;
        y_cur = yp1;
    }
    return y_cur;
}

KOKKOS_INLINE_FUNCTION FloatFloat zeta(FloatFloat s) {
    if (s.hi <= 1.0f) { Kokkos::printf("FFZETA: s <= 1\n"); return FloatFloat(0.0f); }
    const int N = 30;
    FloatFloat sum = FloatFloat(0.0f);
    for (int k = 1; k <= N; ++k)
        sum = add(sum, exp(multiply(negate(s), log(FloatFloat((float)k)))));
    FloatFloat tail = divide(exp(multiply(subtract(FloatFloat(1.0f), s), log(FloatFloat((float)N)))),
                         subtract(s, FloatFloat(1.0f)));
    return add(sum, tail);
}

KOKKOS_INLINE_FUNCTION FloatFloat expint(FloatFloat x) {
    FloatFloat eg = FloatFloat_euler_gamma();
    FloatFloat sum = add(eg, log(abs(x)));
    FloatFloat term = x;
    for (int k = 1; k <= 60; ++k) {
        sum = add(sum, divide_scalar(term, (float)(k * k)));
        term = multiply(term, x);
        if (Kokkos::fabs(term.hi) * 1e-15f < Kokkos::fabs(sum.hi)) break;
    }
    return sum;
}

KOKKOS_INLINE_FUNCTION FloatFloat incgamma(FloatFloat a, FloatFloat x) {
    const float eps = 1.0e-15f;
    FloatFloat term = divide(exp(negate(x)), a);
    FloatFloat sum  = term;
    for (int k = 1; k <= 60; ++k) {
        term = multiply(term, divide(x, add(a, FloatFloat((float)k))));
        sum  = add(sum, term);
        if (Kokkos::fabs(term.hi) < eps * Kokkos::fabs(sum.hi)) break;
    }
    return multiply(sum, exp(multiply(a, log(x))));
}

} // namespace Experimental
} // namespace Kokkos

// ============================================================
// Re-exposure under namespace Kokkos (T0.4/T2.0)
// ============================================================
// Mirrors impl/Kokkos_QuadPrecisionMath.hpp's __float128 overloads (and
// dd_math.hpp's block) so user code can call Kokkos::exp(ff) identically to
// Kokkos::exp(double)/Kokkos::exp(__float128). One-line forwards to the
// Kokkos::Experimental implementations.
// NOTE: add/subtract/multiply/divide are deliberately NOT forwarded here — those
// are reached via operators and explicit ADL, not as Kokkos::add etc.
namespace Kokkos {
// clang-format off
KOKKOS_INLINE_FUNCTION Experimental::FloatFloat abs(Experimental::FloatFloat x)   { return Experimental::abs(x); }
KOKKOS_INLINE_FUNCTION Experimental::FloatFloat sqrt(Experimental::FloatFloat x)  { return Experimental::sqrt(x); }
KOKKOS_INLINE_FUNCTION Experimental::FloatFloat exp(Experimental::FloatFloat x)   { return Experimental::exp(x); }
KOKKOS_INLINE_FUNCTION Experimental::FloatFloat exp2(Experimental::FloatFloat x)  { return Experimental::exp2(x); }
KOKKOS_INLINE_FUNCTION Experimental::FloatFloat exp10(Experimental::FloatFloat x) { return Experimental::exp10(x); }
KOKKOS_INLINE_FUNCTION Experimental::FloatFloat expm1(Experimental::FloatFloat x) { return Experimental::expm1(x); }
KOKKOS_INLINE_FUNCTION Experimental::FloatFloat log(Experimental::FloatFloat x)   { return Experimental::log(x); }
KOKKOS_INLINE_FUNCTION Experimental::FloatFloat log2(Experimental::FloatFloat x)  { return Experimental::log2(x); }
KOKKOS_INLINE_FUNCTION Experimental::FloatFloat log10(Experimental::FloatFloat x) { return Experimental::log10(x); }
KOKKOS_INLINE_FUNCTION Experimental::FloatFloat log1p(Experimental::FloatFloat x) { return Experimental::log1p(x); }
KOKKOS_INLINE_FUNCTION Experimental::FloatFloat sin(Experimental::FloatFloat x)   { return Experimental::sin(x); }
KOKKOS_INLINE_FUNCTION Experimental::FloatFloat cos(Experimental::FloatFloat x)   { return Experimental::cos(x); }
KOKKOS_INLINE_FUNCTION Experimental::FloatFloat tan(Experimental::FloatFloat x)   { return Experimental::tan(x); }
KOKKOS_INLINE_FUNCTION Experimental::FloatFloat asin(Experimental::FloatFloat x)  { return Experimental::asin(x); }
KOKKOS_INLINE_FUNCTION Experimental::FloatFloat acos(Experimental::FloatFloat x)  { return Experimental::acos(x); }
KOKKOS_INLINE_FUNCTION Experimental::FloatFloat atan(Experimental::FloatFloat x)  { return Experimental::atan(x); }
KOKKOS_INLINE_FUNCTION Experimental::FloatFloat atan2(Experimental::FloatFloat y, Experimental::FloatFloat x) { return Experimental::atan2(y, x); }
KOKKOS_INLINE_FUNCTION Experimental::FloatFloat sinh(Experimental::FloatFloat x)  { return Experimental::sinh(x); }
KOKKOS_INLINE_FUNCTION Experimental::FloatFloat cosh(Experimental::FloatFloat x)  { return Experimental::cosh(x); }
KOKKOS_INLINE_FUNCTION Experimental::FloatFloat tanh(Experimental::FloatFloat x)  { return Experimental::tanh(x); }
KOKKOS_INLINE_FUNCTION Experimental::FloatFloat asinh(Experimental::FloatFloat x) { return Experimental::asinh(x); }
KOKKOS_INLINE_FUNCTION Experimental::FloatFloat acosh(Experimental::FloatFloat x) { return Experimental::acosh(x); }
KOKKOS_INLINE_FUNCTION Experimental::FloatFloat atanh(Experimental::FloatFloat x) { return Experimental::atanh(x); }
KOKKOS_INLINE_FUNCTION Experimental::FloatFloat pow(Experimental::FloatFloat a, Experimental::FloatFloat b) { return Experimental::pow(a, b); }
KOKKOS_INLINE_FUNCTION Experimental::FloatFloat hypot(Experimental::FloatFloat a, Experimental::FloatFloat b) { return Experimental::hypot(a, b); }
KOKKOS_INLINE_FUNCTION Experimental::FloatFloat fmod(Experimental::FloatFloat a, Experimental::FloatFloat b) { return Experimental::fmod(a, b); }
KOKKOS_INLINE_FUNCTION Experimental::FloatFloat remainder(Experimental::FloatFloat a, Experimental::FloatFloat b) { return Experimental::remainder(a, b); }
KOKKOS_INLINE_FUNCTION Experimental::FloatFloat copysign(Experimental::FloatFloat a, Experimental::FloatFloat b) { return Experimental::copysign(a, b); }
KOKKOS_INLINE_FUNCTION Experimental::FloatFloat fmax(Experimental::FloatFloat a, Experimental::FloatFloat b) { return Experimental::fmax(a, b); }
KOKKOS_INLINE_FUNCTION Experimental::FloatFloat fmin(Experimental::FloatFloat a, Experimental::FloatFloat b) { return Experimental::fmin(a, b); }
KOKKOS_INLINE_FUNCTION Experimental::FloatFloat fdim(Experimental::FloatFloat a, Experimental::FloatFloat b) { return Experimental::fdim(a, b); }
KOKKOS_INLINE_FUNCTION Experimental::FloatFloat fma(Experimental::FloatFloat a, Experimental::FloatFloat b, Experimental::FloatFloat c) { return Experimental::fma(a, b, c); }
KOKKOS_INLINE_FUNCTION Experimental::FloatFloat ceil(Experimental::FloatFloat x)  { return Experimental::ceil(x); }
KOKKOS_INLINE_FUNCTION Experimental::FloatFloat floor(Experimental::FloatFloat x) { return Experimental::floor(x); }
KOKKOS_INLINE_FUNCTION Experimental::FloatFloat round(Experimental::FloatFloat x) { return Experimental::round(x); }
KOKKOS_INLINE_FUNCTION Experimental::FloatFloat trunc(Experimental::FloatFloat x) { return Experimental::trunc(x); }
KOKKOS_INLINE_FUNCTION Experimental::FloatFloat erf(Experimental::FloatFloat x)   { return Experimental::erf(x); }
KOKKOS_INLINE_FUNCTION Experimental::FloatFloat erfc(Experimental::FloatFloat x)  { return Experimental::erfc(x); }
KOKKOS_INLINE_FUNCTION Experimental::FloatFloat tgamma(Experimental::FloatFloat x){ return Experimental::tgamma(x); }
// clang-format on
}  // namespace Kokkos
