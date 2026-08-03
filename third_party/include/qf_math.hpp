// SPDX-License-Identifier: LicenseRef-LBNL-BSD-License
//
// Copyright (c) 2003-2023 The Regents of the University of California, through
//   Lawrence Berkeley National Laboratory — QD 2.3.24 (original algorithms;
//   Yozo Hida, Xiaoye S. Li, David H. Bailey)
// Modifications Copyright (c) 2026 UChicago Argonne, LLC
//
// This file is a mechanical port of the QD 2.3.24 quad-double package
// (qd/src/qd_real.cpp and qd/include/qd/qd_inline.h) from four-word FP64
// (quad-double, ~212-bit significand) to four-word FP32 (quad-float,
// "QuadFloat", ~96-bit significand, ~29 decimal digits, u = 2^-96). The
// algorithm structure — Priest renormalization (renorm_4), Hida-Li-Bailey
// sloppy/ieee addition, sloppy multiplication, long-division, and Heron
// square-root — descends directly from QD 2.3.24. Every non-trivial routine
// cites its QD source location.
//
// LICENSE LINEAGE (per docs/TEST_SUITE_PLAN.md §"Phase 2/3 open question — FF
// and QF port lineage", T3.0a kickoff): QF is modeled on QD, NOT on DDFUN, so
// it inherits QD's LBNL-BSD-License (triple-authored Hida/Li/Bailey, LBNL
// *institutional* copyright, commercial contact ipo@lbl.gov / TTD@lbl.gov) —
// which is a DIFFERENT license than the DHB-License that governs dd_math.hpp /
// ff_math.hpp (Bailey personal copyright, DDFUN provenance). The header here
// therefore carries LicenseRef-LBNL-BSD-License, not LicenseRef-DHB-License.
// See LICENSES/LicenseRef-LBNL-BSD-License.txt for the full text and NOTICE.md
// for the per-file mapping.
//
// FP32-specific porting notes (splitter reuse, Newton/Heron iteration counts,
// sloppy_add safety at the narrower FP32 exponent, constant generation) are
// documented in docs/PORT_NOTES_QF.md.

#pragma once

// Quad-float real arithmetic — Kokkos::Experimental::QuadFloat.
// All functions KOKKOS_INLINE_FUNCTION (host + device via Kokkos/CUDA).
// Ported from QD 2.3.24 (Hida-Li-Bailey), swapping 4×FP64 for 4×FP32. Reuses
// the FP32 error-free-transform primitives (Knuth twoSum, Dekker twoProduct)
// already validated for the FF backend in third_party/include/ff_math.hpp
// (T2.1); the QD-style by-reference forms below are bit-identical to those.
//
// Precision: ~28.9 decimal digits (24-bit FP32 mantissa × 4 = ~96 bits).
// Range: bounded by FP32 (~3.4e38), much tighter than FP64.
//
// Naming conventions (T0.4/T2.0, for eventual upstreaming to Kokkos) mirror
// dd_math.hpp / ff_math.hpp:
//   * Type + math live under namespace Kokkos::Experimental.
//   * Arithmetic free functions use STL-style names (add/subtract/multiply/
//     divide/negate) and are also reachable through operator overloads.
//   * Constants are free functions QuadFloat_pi(), QuadFloat_e(), ...
//   * The bit-pattern constructor is the static factory
//     QuadFloat::from_bits(f0, f1, f2, f3).
//   * Single-output math functions are re-exposed under namespace Kokkos at the
//     bottom of this header (T3.0a exposes abs/sqrt; T3.0b adds the
//     transcendentals). add/subtract/multiply/divide are NOT re-exposed under
//     Kokkos — operators + explicit ADL only, same posture as DD/FF.
//
// SCOPE (T3.0a): arithmetic + renormalization only — renorm_4, add (sloppy +
// ieee), subtract, negate, abs, multiply, divide, sqrt, round_to_nearest_int,
// pow_int, and the six math constants. Transcendentals (exp/log/sin/...) are
// T3.0b. No qf_complex.hpp.

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
struct QuadFloat;
KOKKOS_INLINE_FUNCTION QuadFloat add(QuadFloat a, QuadFloat b);
KOKKOS_INLINE_FUNCTION QuadFloat subtract(QuadFloat a, QuadFloat b);
KOKKOS_INLINE_FUNCTION QuadFloat multiply(QuadFloat a, QuadFloat b);
KOKKOS_INLINE_FUNCTION QuadFloat divide(QuadFloat a, QuadFloat b);
KOKKOS_INLINE_FUNCTION QuadFloat multiply_scalar(QuadFloat a, float b);
KOKKOS_INLINE_FUNCTION QuadFloat mul_pwr2(QuadFloat a, float b);
KOKKOS_INLINE_FUNCTION QuadFloat negate(QuadFloat a);
KOKKOS_INLINE_FUNCTION QuadFloat abs(QuadFloat a);
KOKKOS_INLINE_FUNCTION QuadFloat sqr(QuadFloat a);
KOKKOS_INLINE_FUNCTION QuadFloat sqrt(QuadFloat a);
KOKKOS_INLINE_FUNCTION QuadFloat round_to_nearest_int(QuadFloat a);
KOKKOS_INLINE_FUNCTION QuadFloat pow_int(QuadFloat a, int n);

// ============================================================
// Error-free transforms (FP32).
// Bit-identical to the primitives validated for FF in ff_math.hpp (T2.1);
// expressed here in QD's by-reference form (QD 2.3.24 qd/include/qd/inline.h).
// ============================================================

// fl(a+b) and err, assuming |a| >= |b|.  QD inline.h:35-39 (quick_two_sum).
KOKKOS_INLINE_FUNCTION float qf_quick_two_sum(float a, float b, float& err) {
    float s = a + b;
    err = b - (s - a);
    return s;
}

// fl(a+b) and err (Knuth TwoSum, no ordering assumption).  QD inline.h:49-55.
// Mirror of the twoSum inside ff_math.hpp add() (ff_math.hpp:174-181).
KOKKOS_INLINE_FUNCTION float qf_two_sum(float a, float b, float& err) {
    float s  = a + b;
    float bb = s - a;
    err = (a - (s - bb)) + (b - bb);
    return s;
}

// fl(a*b) and err (Dekker TwoProduct via Veltkamp split).  QD inline.h:85-99.
// Splitter 8193.0f = 2^13+1 for the 24-bit FP32 mantissa — the same constant
// used and validated for FF (ff_math.hpp:194, two_prod ff_math.hpp:266-274).
// Empirically exact over |operands| <= 1e6 (scripts/gen_qf_constants harness /
// scripts/test_qfmul.cpp). Large-magnitude splitter overflow (QD's
// _QD_SPLIT_THRESH branch, inline.h:66-83) is NOT ported — see PORT_NOTES_QF.
KOKKOS_INLINE_FUNCTION float qf_two_prod(float a, float b, float& err) {
    const float split = 8193.0f;
    float cona = a * split, conb = b * split;
    float a1 = cona - (cona - a), b1 = conb - (conb - b);
    float a2 = a - a1,            b2 = b - b1;
    float p  = a * b;
    err = ((a1 * b1 - p) + a1 * b2 + a2 * b1) + a2 * b2;
    return p;
}

// fl(a*a) and err.  QD inline.h:101-113 (two_sqr).
KOKKOS_INLINE_FUNCTION float qf_two_sqr(float a, float& err) {
    const float split = 8193.0f;
    float con = a * split;
    float hi  = con - (con - a);
    float lo  = a - hi;
    float q   = a * a;
    err = ((hi * hi - q) + 2.0f * hi * lo) + lo * lo;
    return q;
}

// three_sum / three_sum2.  QD inline.h:192-204.
KOKKOS_INLINE_FUNCTION void qf_three_sum(float& a, float& b, float& c) {
    float t1, t2, t3;
    t1 = qf_two_sum(a, b, t2);
    a  = qf_two_sum(c, t1, t3);
    b  = qf_two_sum(t2, t3, c);
}
KOKKOS_INLINE_FUNCTION void qf_three_sum2(float& a, float& b, float& c) {
    float t1, t2, t3;
    t1 = qf_two_sum(a, b, t2);
    a  = qf_two_sum(c, t1, t3);
    b  = t2 + t3;
}

// ============================================================
// Renormalization (Priest normalization — Hida-Li-Bailey Algorithm 3)
// ============================================================

// Length-4 renormalization: collapse a 4-word unnormalized expansion to a
// non-overlapping length-4 QuadFloat.  Port of qd::renorm(c0,c1,c2,c3),
// QD 2.3.24 qd/include/qd/qd_inline.h:95-125.  Used by divide() and
// round_to_nearest_int().
KOKKOS_INLINE_FUNCTION void renorm(float& c0, float& c1, float& c2, float& c3) {
    float s0, s1, s2 = 0.0f, s3 = 0.0f;
    if (Kokkos::isinf(c0)) return;

    s0 = qf_quick_two_sum(c2, c3, c3);
    s0 = qf_quick_two_sum(c1, s0, c2);
    c0 = qf_quick_two_sum(c0, s0, c1);

    s0 = c0;
    s1 = c1;
    if (s1 != 0.0f) {
        s1 = qf_quick_two_sum(s1, c2, s2);
        if (s2 != 0.0f)
            s2 = qf_quick_two_sum(s2, c3, s3);
        else
            s1 = qf_quick_two_sum(s1, c3, s2);
    } else {
        s0 = qf_quick_two_sum(s0, c2, s1);
        if (s1 != 0.0f)
            s1 = qf_quick_two_sum(s1, c3, s2);
        else
            s0 = qf_quick_two_sum(s0, c3, s1);
    }
    c0 = s0; c1 = s1; c2 = s2; c3 = s3;
}

// Length-5 -> length-4 renormalization (the task's "renorm_4"): collapse a
// 5-word unnormalized accumulator (the natural output width of add/multiply/
// multiply_scalar/divide) to a non-overlapping length-4 QuadFloat.  Port of
// qd::renorm(c0,c1,c2,c3,c4), QD 2.3.24 qd_inline.h:127-177.
KOKKOS_INLINE_FUNCTION void renorm_4(float& c0, float& c1, float& c2,
                                     float& c3, float& c4) {
    float s0, s1, s2 = 0.0f, s3 = 0.0f;
    if (Kokkos::isinf(c0)) return;

    s0 = qf_quick_two_sum(c3, c4, c4);
    s0 = qf_quick_two_sum(c2, s0, c3);
    s0 = qf_quick_two_sum(c1, s0, c2);
    c0 = qf_quick_two_sum(c0, s0, c1);

    s0 = c0;
    s1 = c1;

    if (s1 != 0.0f) {
        s1 = qf_quick_two_sum(s1, c2, s2);
        if (s2 != 0.0f) {
            s2 = qf_quick_two_sum(s2, c3, s3);
            if (s3 != 0.0f)
                s3 += c4;
            else
                s2 = qf_quick_two_sum(s2, c4, s3);
        } else {
            s1 = qf_quick_two_sum(s1, c3, s2);
            if (s2 != 0.0f)
                s2 = qf_quick_two_sum(s2, c4, s3);
            else
                s1 = qf_quick_two_sum(s1, c4, s2);
        }
    } else {
        s0 = qf_quick_two_sum(s0, c2, s1);
        if (s1 != 0.0f) {
            s1 = qf_quick_two_sum(s1, c3, s2);
            if (s2 != 0.0f)
                s2 = qf_quick_two_sum(s2, c4, s3);
            else
                s1 = qf_quick_two_sum(s1, c4, s2);
        } else {
            s0 = qf_quick_two_sum(s0, c3, s1);
            if (s1 != 0.0f)
                s1 = qf_quick_two_sum(s1, c4, s2);
            else
                s0 = qf_quick_two_sum(s0, c4, s1);
        }
    }
    c0 = s0; c1 = s1; c2 = s2; c3 = s3;
}

// ============================================================
// QuadFloat struct
// ============================================================
struct QuadFloat {
    float f0, f1, f2, f3;

    KOKKOS_INLINE_FUNCTION QuadFloat() : f0(0.0f), f1(0.0f), f2(0.0f), f3(0.0f) {}
    KOKKOS_INLINE_FUNCTION QuadFloat(float x) : f0(x), f1(0.0f), f2(0.0f), f3(0.0f) {}
    KOKKOS_INLINE_FUNCTION QuadFloat(float a0, float a1, float a2, float a3)
        : f0(a0), f1(a1), f2(a2), f3(a3) {}

    // Faithfully encode an FP64 value by successive FP32 splitting (Route-A,
    // length-4 analogue of ff_math.hpp's ffloat(double)). A double carries 53
    // bits, so two words suffice; the remaining words fall to 0 after the split.
    KOKKOS_INLINE_FUNCTION QuadFloat(double x) {
        double r = x;
        float  c0 = (float)r; r -= (double)c0;
        float  c1 = (float)r; r -= (double)c1;
        float  c2 = (float)r; r -= (double)c2;
        float  c3 = (float)r;
        f0 = c0; f1 = c1; f2 = c2; f3 = c3;
    }

    KOKKOS_INLINE_FUNCTION QuadFloat(const QuadFloat& o)
        : f0(o.f0), f1(o.f1), f2(o.f2), f3(o.f3) {}
    KOKKOS_INLINE_FUNCTION QuadFloat& operator=(const QuadFloat& o) {
        f0=o.f0; f1=o.f1; f2=o.f2; f3=o.f3; return *this;
    }

    KOKKOS_INLINE_FUNCTION float operator[](int i) const {
        return (i==0)?f0:(i==1)?f1:(i==2)?f2:f3;
    }

    // Factory: build a QuadFloat from the IEEE-754 bit patterns of its four
    // FP32 components. Safe on host (memcpy) and device (__int_as_float).
    static KOKKOS_INLINE_FUNCTION QuadFloat from_bits(uint32_t b0, uint32_t b1,
                                                      uint32_t b2, uint32_t b3) {
        float a0, a1, a2, a3;
#ifndef __CUDA_ARCH__
        std::memcpy(&a0, &b0, sizeof(float));
        std::memcpy(&a1, &b1, sizeof(float));
        std::memcpy(&a2, &b2, sizeof(float));
        std::memcpy(&a3, &b3, sizeof(float));
#else
        a0 = __int_as_float(static_cast<int>(b0));
        a1 = __int_as_float(static_cast<int>(b1));
        a2 = __int_as_float(static_cast<int>(b2));
        a3 = __int_as_float(static_cast<int>(b3));
#endif
        return QuadFloat(a0, a1, a2, a3);
    }

    KOKKOS_INLINE_FUNCTION QuadFloat operator-() const { return negate(*this); }
    KOKKOS_INLINE_FUNCTION QuadFloat operator+(QuadFloat b) const { return add(*this, b); }
    KOKKOS_INLINE_FUNCTION QuadFloat operator-(QuadFloat b) const { return subtract(*this, b); }
    KOKKOS_INLINE_FUNCTION QuadFloat operator*(QuadFloat b) const { return multiply(*this, b); }
    KOKKOS_INLINE_FUNCTION QuadFloat operator/(QuadFloat b) const { return divide(*this, b); }
    KOKKOS_INLINE_FUNCTION QuadFloat operator*(float b) const { return multiply_scalar(*this, b); }
    KOKKOS_INLINE_FUNCTION QuadFloat operator+(float b) const { return add(*this, QuadFloat(b)); }
    KOKKOS_INLINE_FUNCTION QuadFloat operator-(float b) const { return subtract(*this, QuadFloat(b)); }

    KOKKOS_INLINE_FUNCTION QuadFloat& operator+=(QuadFloat b) { *this = *this + b; return *this; }
    KOKKOS_INLINE_FUNCTION QuadFloat& operator-=(QuadFloat b) { *this = *this - b; return *this; }
    KOKKOS_INLINE_FUNCTION QuadFloat& operator*=(QuadFloat b) { *this = *this * b; return *this; }
    KOKKOS_INLINE_FUNCTION QuadFloat& operator/=(QuadFloat b) { *this = *this / b; return *this; }

    KOKKOS_INLINE_FUNCTION bool operator==(QuadFloat b) const {
        return f0==b.f0 && f1==b.f1 && f2==b.f2 && f3==b.f3;
    }
    KOKKOS_INLINE_FUNCTION bool operator!=(QuadFloat b) const { return !(*this == b); }
    KOKKOS_INLINE_FUNCTION bool operator<(QuadFloat b) const {
        if (f0 != b.f0) return f0 < b.f0;
        if (f1 != b.f1) return f1 < b.f1;
        if (f2 != b.f2) return f2 < b.f2;
        return f3 < b.f3;
    }
    KOKKOS_INLINE_FUNCTION bool operator>(QuadFloat b) const {
        if (f0 != b.f0) return f0 > b.f0;
        if (f1 != b.f1) return f1 > b.f1;
        if (f2 != b.f2) return f2 > b.f2;
        return f3 > b.f3;
    }
    KOKKOS_INLINE_FUNCTION bool operator<=(QuadFloat b) const { return !(*this > b); }
    KOKKOS_INLINE_FUNCTION bool operator>=(QuadFloat b) const { return !(*this < b); }
};

KOKKOS_INLINE_FUNCTION QuadFloat operator+(float a, QuadFloat b) { return add(QuadFloat(a), b); }
KOKKOS_INLINE_FUNCTION QuadFloat operator-(float a, QuadFloat b) { return subtract(QuadFloat(a), b); }
KOKKOS_INLINE_FUNCTION QuadFloat operator*(float a, QuadFloat b) { return multiply_scalar(b, a); }

#ifndef __CUDA_ARCH__
inline std::ostream& operator<<(std::ostream& os, const QuadFloat& d) {
    os << "[" << std::setprecision(8) << std::scientific
       << d.f0 << ", " << d.f1 << ", " << d.f2 << ", " << d.f3 << "]";
    return os;
}
#endif

// ============================================================
// Constants (4×FP32 bit patterns).
// Auto-generated by scripts/gen_qf_constants.cpp — do not edit by hand.
// Successive-splitting of a 113-bit __float128 constant into 4 FP32 words;
// reconstruction rel_err < 6e-31 for every entry (well below u = 2^-96).
// ============================================================
KOKKOS_INLINE_FUNCTION QuadFloat QuadFloat_pi          () { return QuadFloat::from_bits(0x40490fdbU, 0xb3bbbd2eU, 0xa7772cedU, 0x19cc5170U); } // pi
KOKKOS_INLINE_FUNCTION QuadFloat QuadFloat_e           () { return QuadFloat::from_bits(0x402df854U, 0x33b14577U, 0xa7559541U, 0x1ae2b101U); } // e
KOKKOS_INLINE_FUNCTION QuadFloat QuadFloat_log2        () { return QuadFloat::from_bits(0x3f317218U, 0xb102e308U, 0xa4ca86c4U, 0x186ce601U); } // ln(2)
KOKKOS_INLINE_FUNCTION QuadFloat QuadFloat_log10       () { return QuadFloat::from_bits(0x40135d8eU, 0xb309555dU, 0xa69f48adU, 0x9a129d48U); } // ln(10)
KOKKOS_INLINE_FUNCTION QuadFloat QuadFloat_sqrt2       () { return QuadFloat::from_bits(0x3fb504f3U, 0x32cfe77aU, 0xa65bdd34U, 0x989d9323U); } // sqrt(2)
KOKKOS_INLINE_FUNCTION QuadFloat QuadFloat_euler_gamma () { return QuadFloat::from_bits(0x3f13c468U, 0xb1e4127aU, 0x24f49a38U, 0x97e03f7fU); } // Euler gamma

// ============================================================
// Negation / absolute value
// ============================================================

// QD qd_inline.h:438-440 (operator-).
KOKKOS_INLINE_FUNCTION QuadFloat negate(QuadFloat a) {
    return QuadFloat(-a.f0, -a.f1, -a.f2, -a.f3);
}

// QD qd_inline.h:788-790 (abs).
KOKKOS_INLINE_FUNCTION QuadFloat abs(QuadFloat a) {
    return (a.f0 < 0.0f) ? negate(a) : a;
}

// ============================================================
// Addition
// ============================================================

// IEEE-style addition (satisfies the IEEE error bound).  Port of
// qd_real::ieee_add, QD 2.3.24 qd_inline.h:286-336, plus the quick_three_accum
// helper it calls (qd_inline.h:261-282).  NOT the default (QD builds with
// QD_IEEE_ADD off by default); provided for parity with the DD/FF story and for
// callers that need the tighter bound.
KOKKOS_INLINE_FUNCTION float qf_quick_three_accum(float& a, float& b, float c) {
    float s;
    bool za, zb;
    s = qf_two_sum(b, c, b);
    s = qf_two_sum(a, s, a);
    za = (a != 0.0f);
    zb = (b != 0.0f);
    if (za && zb) return s;
    if (!zb) { b = a; a = s; }
    else     { a = s; }
    return 0.0f;
}

KOKKOS_INLINE_FUNCTION QuadFloat ieee_add(QuadFloat a, QuadFloat b) {
    int i, j, k;
    float s, t;
    float u, v;
    float x[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    i = j = k = 0;
    if (Kokkos::fabs(a[i]) > Kokkos::fabs(b[j])) u = a[i++]; else u = b[j++];
    if (Kokkos::fabs(a[i]) > Kokkos::fabs(b[j])) v = a[i++]; else v = b[j++];

    u = qf_quick_two_sum(u, v, v);

    while (k < 4) {
        if (i >= 4 && j >= 4) {
            x[k] = u;
            if (k < 3) x[++k] = v;
            break;
        }
        if (i >= 4)                                    t = b[j++];
        else if (j >= 4)                               t = a[i++];
        else if (Kokkos::fabs(a[i]) > Kokkos::fabs(b[j])) t = a[i++];
        else                                           t = b[j++];

        s = qf_quick_three_accum(u, v, t);
        if (s != 0.0f) x[k++] = s;
    }
    for (k = i; k < 4; k++) x[3] += a[k];
    for (k = j; k < 4; k++) x[3] += b[k];

    renorm(x[0], x[1], x[2], x[3]);
    return QuadFloat(x[0], x[1], x[2], x[3]);
}

// Cray-style ("sloppy") addition — the QD DEFAULT (QD_IEEE_ADD undefined).
// Port of qd_real::sloppy_add, QD 2.3.24 qd_inline.h:338-405 (the active,
// data-dependency-minimized form; QD documents it as identical to the
// commented two_sum version at qd_inline.h:339-354).  This is what
// add()/operator+ dispatch to, matching QD's out-of-the-box configuration.
// Safe at FP32: correctness rests on the inputs being non-overlapping
// expansions, not on the exponent range (see PORT_NOTES_QF).
KOKKOS_INLINE_FUNCTION QuadFloat sloppy_add(QuadFloat a, QuadFloat b) {
    float s0, s1, s2, s3;
    float t0, t1, t2, t3;
    float v0, v1, v2, v3;
    float u0, u1, u2, u3;
    float w0, w1, w2, w3;

    s0 = a.f0 + b.f0;  s1 = a.f1 + b.f1;  s2 = a.f2 + b.f2;  s3 = a.f3 + b.f3;

    v0 = s0 - a.f0;    v1 = s1 - a.f1;    v2 = s2 - a.f2;    v3 = s3 - a.f3;
    u0 = s0 - v0;      u1 = s1 - v1;      u2 = s2 - v2;      u3 = s3 - v3;
    w0 = a.f0 - u0;    w1 = a.f1 - u1;    w2 = a.f2 - u2;    w3 = a.f3 - u3;

    u0 = b.f0 - v0;    u1 = b.f1 - v1;    u2 = b.f2 - v2;    u3 = b.f3 - v3;

    t0 = w0 + u0;      t1 = w1 + u1;      t2 = w2 + u2;      t3 = w3 + u3;

    s1 = qf_two_sum(s1, t0, t0);
    qf_three_sum(s2, t0, t1);
    qf_three_sum2(s3, t0, t2);
    t0 = t0 + t1 + t3;

    renorm_4(s0, s1, s2, s3, t0);
    return QuadFloat(s0, s1, s2, s3);
}

// QD default dispatch (qd_inline.h:408-414): operator+ -> sloppy_add.
KOKKOS_INLINE_FUNCTION QuadFloat add(QuadFloat a, QuadFloat b) {
    return sloppy_add(a, b);
}

// QD qd_inline.h:459-461 (operator-): a - b == a + (-b).
KOKKOS_INLINE_FUNCTION QuadFloat subtract(QuadFloat a, QuadFloat b) {
    return add(a, negate(b));
}

// ============================================================
// Multiplication
// ============================================================

// quad-float * float.  Port of operator*(qd_real, double),
// QD 2.3.24 qd_inline.h:490-514.
KOKKOS_INLINE_FUNCTION QuadFloat multiply_scalar(QuadFloat a, float b) {
    float p0, p1, p2, p3;
    float q0, q1, q2;
    float s0, s1, s2, s3, s4;

    p0 = qf_two_prod(a.f0, b, q0);
    p1 = qf_two_prod(a.f1, b, q1);
    p2 = qf_two_prod(a.f2, b, q2);
    p3 = a.f3 * b;

    s0 = p0;
    s1 = qf_two_sum(q0, p1, s2);
    qf_three_sum(s2, q1, p2);
    qf_three_sum2(q1, q2, p3);
    s3 = q1;
    s4 = q2 + p2;

    renorm_4(s0, s1, s2, s3, s4);
    return QuadFloat(s0, s1, s2, s3);
}

// Exact multiplication by a power of two (no rounding).  QD qd_inline.h:485-487.
KOKKOS_INLINE_FUNCTION QuadFloat mul_pwr2(QuadFloat a, float b) {
    return QuadFloat(a.f0 * b, a.f1 * b, a.f2 * b, a.f3 * b);
}

// quad-float * quad-float — "sloppy" multiplication, the QD DEFAULT
// (QD builds with QD_SLOPPY_MUL on).  16 partial products a_i*b_j; the leading
// 6 (weights u^0..u^2) are formed with exact two_prod, the O(u^3) terms are
// folded in scalar, and terms of weight u^4 and higher are dropped before
// renorm_4 collapses the length-5 accumulator to length-4.  Port of
// qd_real::sloppy_mul, QD 2.3.24 qd_inline.h:567-599.
KOKKOS_INLINE_FUNCTION QuadFloat multiply(QuadFloat a, QuadFloat b) {
    float p0, p1, p2, p3, p4, p5;
    float q0, q1, q2, q3, q4, q5;
    float t0, t1;
    float s0, s1, s2;

    p0 = qf_two_prod(a.f0, b.f0, q0);

    p1 = qf_two_prod(a.f0, b.f1, q1);
    p2 = qf_two_prod(a.f1, b.f0, q2);

    p3 = qf_two_prod(a.f0, b.f2, q3);
    p4 = qf_two_prod(a.f1, b.f1, q4);
    p5 = qf_two_prod(a.f2, b.f0, q5);

    // Start accumulation.
    qf_three_sum(p1, p2, q0);

    // Six-three sum of (p2, q1, q2) and (p3, p4, p5).
    qf_three_sum(p2, q1, q2);
    qf_three_sum(p3, p4, p5);
    s0 = qf_two_sum(p2, p3, t0);
    s1 = qf_two_sum(q1, p4, t1);
    s2 = q2 + p5;
    s1 = qf_two_sum(s1, t0, t0);
    s2 += (t0 + t1);

    // O(u^3) terms: the nine remaining cross-products, folded in scalar.
    s1 += a.f0*b.f3 + a.f1*b.f2 + a.f2*b.f1 + a.f3*b.f0 + q0 + q3 + q4 + q5;
    renorm_4(p0, p1, s0, s1, s2);
    return QuadFloat(p0, p1, s0, s1);
}

// quad-float ^ 2.  Port of sqr(qd_real), QD 2.3.24 qd_inline.h:674-715.
KOKKOS_INLINE_FUNCTION QuadFloat sqr(QuadFloat a) {
    float p0, p1, p2, p3, p4, p5;
    float q0, q1, q2, q3;
    float s0, s1;
    float t0, t1;

    p0 = qf_two_sqr(a.f0, q0);
    p1 = qf_two_prod(2.0f * a.f0, a.f1, q1);
    p2 = qf_two_prod(2.0f * a.f0, a.f2, q2);
    p3 = qf_two_sqr(a.f1, q3);

    p1 = qf_two_sum(q0, p1, q0);

    q0 = qf_two_sum(q0, q1, q1);
    p2 = qf_two_sum(p2, p3, p3);

    s0 = qf_two_sum(q0, p2, t0);
    s1 = qf_two_sum(q1, p3, t1);

    s1 = qf_two_sum(s1, t0, t0);
    t0 += t1;

    s1 = qf_quick_two_sum(s1, t0, t0);
    p2 = qf_quick_two_sum(s0, s1, t1);
    p3 = qf_quick_two_sum(t1, t0, q0);

    p4 = 2.0f * a.f0 * a.f3;
    p5 = 2.0f * a.f1 * a.f2;

    p4 = qf_two_sum(p4, p5, p5);
    q2 = qf_two_sum(q2, q3, q3);

    t0 = qf_two_sum(p4, q2, t1);
    t1 = t1 + p5 + q3;

    p3 = qf_two_sum(p3, t0, p4);
    p4 = p4 + q0 + t1;

    renorm_4(p0, p1, p2, p3, p4);
    return QuadFloat(p0, p1, p2, p3);
}

// ============================================================
// Division
// ============================================================

// quad-float / quad-float — "sloppy" long division, the QD DEFAULT
// (QD builds with QD_SLOPPY_DIV on).  Port of qd_real::sloppy_div,
// QD 2.3.24 qd/src/qd_real.cpp:693-712.
//
// NOTE ON ALGORITHM (source-fidelity, rule 6): the T3.0a task text describes
// divide as "Newton iteration, initial reciprocal from FP32 division, 3
// iterations". QD 2.3.24's qd_real::div is NOT Newton — it is classical long
// division: each quotient digit q_k = r[0]/b[0] contributes ~24 fresh bits
// (q0~24, q1~48, q2~72, q3~96), and the residual r is refined by a full
// QuadFloat multiply-subtract between digits. Four digits reach the ~96-bit
// QuadFloat width; the accurate variant adds a fifth digit + length-5 renorm.
// This header ports QD's actual routine and cites it; the discrepancy with the
// task text is recorded in PORT_NOTES_QF and the T3.0a report.
KOKKOS_INLINE_FUNCTION QuadFloat divide(QuadFloat a, QuadFloat b) {
    float q0, q1, q2, q3;
    QuadFloat r;

    q0 = a.f0 / b.f0;
    r = subtract(a, multiply_scalar(b, q0));

    q1 = r.f0 / b.f0;
    r = subtract(r, multiply_scalar(b, q1));

    q2 = r.f0 / b.f0;
    r = subtract(r, multiply_scalar(b, q2));

    q3 = r.f0 / b.f0;

    renorm(q0, q1, q2, q3);
    return QuadFloat(q0, q1, q2, q3);
}

// Accurate long division (five quotient digits + length-5 renorm).  Port of
// qd_real::accurate_div, QD 2.3.24 qd_real.cpp:714-736. Not the default;
// provided for parity with QD and for tight-bound callers (T3.4).
KOKKOS_INLINE_FUNCTION QuadFloat divide_accurate(QuadFloat a, QuadFloat b) {
    float q0, q1, q2, q3, q4;
    QuadFloat r;

    q0 = a.f0 / b.f0;  r = subtract(a, multiply_scalar(b, q0));
    q1 = r.f0 / b.f0;  r = subtract(r, multiply_scalar(b, q1));
    q2 = r.f0 / b.f0;  r = subtract(r, multiply_scalar(b, q2));
    q3 = r.f0 / b.f0;  r = subtract(r, multiply_scalar(b, q3));
    q4 = r.f0 / b.f0;

    renorm_4(q0, q1, q2, q3, q4);
    return QuadFloat(q0, q1, q2, q3);
}

// ============================================================
// Square root
// ============================================================

// QuadFloat square root — Heron's method (a Newton iteration on x^2 - a),
// each step doubling the number of correct digits: y = (1/2)(x + a/x).
// Port of fsqrt / sqrt(qd_real), QD 2.3.24 qd/src/qd_real.cpp:738-785.
//
// NOTE ON ALGORITHM (source-fidelity, rule 6): the T3.0a task text describes
// sqrt as "Newton iteration, initial reciprocal from FP32 division, same
// posture as divide" — that is the Karp reciprocal-Newton variant used by
// dd_real::sqrt (QD dd_real.cpp:47-72). QD 2.3.24's qd_real::sqrt is instead
// Heron's method (fsqrt), which needs a full QuadFloat divide per step. This
// header ports QD's actual qd_real::sqrt. Iteration count: the FP32 seed
// sqrt(a.f0) is accurate to ~24 bits; Heron doubles precision per step
// (24 -> 48 -> 96, saturating at the ~96-bit QuadFloat width), so 3 iterations
// suffice. QD's loop runs up to 10 with an early-out convergence test; the port
// keeps that structure with eps = 2^-96 so it stops after ~3 on real inputs.
KOKKOS_INLINE_FUNCTION QuadFloat sqrt(QuadFloat a) {
    if (a.f0 == 0.0f && a.f1 == 0.0f && a.f2 == 0.0f && a.f3 == 0.0f)
        return QuadFloat(0.0f);
    if (a.f0 < 0.0f) {
        Kokkos::printf("QFSQRT: negative argument\n");
        return QuadFloat(0.0f);
    }

    const float eps = 1.2621774e-29f; // ~= 2^-96, QuadFloat unit roundoff
    const QuadFloat half(0.5f);

    QuadFloat x = QuadFloat(Kokkos::sqrt(a.f0));  // ~24-bit FP32 seed
    for (int i = 0; i < 10; ++i) {
        QuadFloat y    = multiply(half, add(x, divide(a, x)));
        QuadFloat diff = subtract(x, y);
        x = y;
        float e = Kokkos::fabs(((diff.f3 + diff.f2) + diff.f1) + diff.f0);
        if (e < Kokkos::fabs(x.f0) * eps)
            return x;
    }
    return x;
}

// ============================================================
// Nearest integer
// ============================================================

// Nearest FP32-int of a single float (round-half-away-from-zero via floor).
// QD 2.3.24 qd/include/qd/inline.h:116-120 (nint(double)), at FP32.
// NOTE (PORT_NOTES_QF): QD's nint does NOT use the 2^(2p-1) magic-constant
// trick that broke FF's ffnint at FP32 (PORT_NOTES §4b); it uses floor(d+0.5),
// which is well-conditioned at every FP32 magnitude, so the FF bug does not
// recur here.
KOKKOS_INLINE_FUNCTION float qf_nint(float d) {
    if (d == Kokkos::floor(d)) return d;
    return Kokkos::floor(d + 0.5f);
}

// Nearest integer of a QuadFloat, component-wise with half-integer tie
// corrections and a final renorm.  Port of nint(qd_real),
// QD 2.3.24 qd/src/qd_real.cpp:48-86.
KOKKOS_INLINE_FUNCTION QuadFloat round_to_nearest_int(QuadFloat a) {
    float x0, x1, x2, x3;
    x0 = qf_nint(a.f0);
    x1 = x2 = x3 = 0.0f;

    if (x0 == a.f0) {
        x1 = qf_nint(a.f1);
        if (x1 == a.f1) {
            x2 = qf_nint(a.f2);
            if (x2 == a.f2) {
                x3 = qf_nint(a.f3);
            } else {
                if (Kokkos::fabs(x2 - a.f2) == 0.5f && a.f3 < 0.0f) x2 -= 1.0f;
            }
        } else {
            if (Kokkos::fabs(x1 - a.f1) == 0.5f && a.f2 < 0.0f) x1 -= 1.0f;
        }
    } else {
        if (Kokkos::fabs(x0 - a.f0) == 0.5f && a.f1 < 0.0f) x0 -= 1.0f;
    }

    renorm(x0, x1, x2, x3);
    return QuadFloat(x0, x1, x2, x3);
}

// ============================================================
// Integer power
// ============================================================

// a^n by binary exponentiation using sqr.  Port of pow(qd_real, int),
// QD 2.3.24 qd/src/qd_real.cpp:568-598.
KOKKOS_INLINE_FUNCTION QuadFloat pow_int(QuadFloat a, int n) {
    if (n == 0) return QuadFloat(1.0f);

    QuadFloat r = a;              // odd-case multiplier
    QuadFloat s = QuadFloat(1.0f); // running answer
    int N = (n < 0) ? -n : n;

    if (N > 1) {
        while (N > 0) {
            if (N % 2 == 1) s = multiply(s, r);
            N /= 2;
            if (N > 0) r = sqr(r);
        }
    } else {
        s = r;
    }

    if (n < 0) return divide(QuadFloat(1.0f), s);
    return s;
}

} // namespace Experimental
} // namespace Kokkos

// ============================================================
// Re-exposure under namespace Kokkos (T0.4/T2.0 convention)
// ============================================================
// Mirrors impl/Kokkos_QuadPrecisionMath.hpp / dd_math.hpp / ff_math.hpp so
// user code can call Kokkos::sqrt(qf) identically to Kokkos::sqrt(double).
// T3.0a exposes the arithmetic-tier single-output functions only; T3.0b adds
// the transcendentals. add/subtract/multiply/divide are deliberately NOT
// forwarded (operators + explicit ADL only, same posture as DD/FF).
namespace Kokkos {
// clang-format off
KOKKOS_INLINE_FUNCTION Experimental::QuadFloat abs(Experimental::QuadFloat x)  { return Experimental::abs(x); }
KOKKOS_INLINE_FUNCTION Experimental::QuadFloat sqrt(Experimental::QuadFloat x) { return Experimental::sqrt(x); }
// clang-format on
}  // namespace Kokkos
