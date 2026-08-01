// SPDX-License-Identifier: LicenseRef-DHB-License
// SPDX-FileCopyrightText: Copyright (c) 2024 David H. Bailey
// SPDX-FileCopyrightText: Modifications Copyright (c) 2026 UChicago Argonne, LLC
//
// Ported from DDFUN v04:
//   https://www.davidhbailey.com/dhbsoftware/ddfun-v04.tar.gz
//   Original author: David H. Bailey (LBNL retired / UC Davis)
//   Original license: DHB-License (modified BSD-3-Clause with §3
//     grant-back clause). Full text: LICENSES/LicenseRef-DHB-License.txt
//     or https://www.davidhbailey.com/dhbsoftware/DHB-License.txt.
//
// This C++/Kokkos port is a derivative work distributed under the
// same DHB-License. See §3 of that license regarding upstream
// contribution rights.
//
// Modifications from the original DDFUN v04 sources:
//   * Translated from Fortran-90 (ddfuna.f90, ddfune.f90) to
//     header-only C++17.
//   * Adapted to Kokkos: every function KOKKOS_INLINE_FUNCTION for
//     host + device portability across CUDA/HIP/SYCL/OpenMP-target.
//   * Namespaced as Kokkos::Experimental::DoubleDouble with STL-
//     style free-function names and ADL-friendly re-exposure under
//     namespace Kokkos, for potential upstreaming to Kokkos.
//   * See docs/TEST_SUITE_PLAN.md "Upstreaming considerations" for
//     naming and API conventions.

#pragma once

// Double-double real arithmetic — Kokkos::Experimental::DoubleDouble.
// All functions KOKKOS_INLINE_FUNCTION (host + device via Kokkos/CUDA).
//
// Ported from DDFUN (David H. Bailey, Lawrence Berkeley National Lab) Fortran
// sources (ddfuna.f90, ddfune.f90).
//
// Naming conventions (T0.4, for eventual upstreaming to Kokkos):
//   * Type + math live under namespace Kokkos::Experimental so an upstream PR is
//     a mechanical move rather than a rewrite.
//   * Arithmetic free functions use STL-style names (add/subtract/multiply/
//     divide/negate) and are also reachable through operator overloads.
//   * Constants are free functions DoubleDouble_pi(), DoubleDouble_e(), ...
//     Chosen over a constants::pi<DoubleDouble>() template because it mirrors
//     Kokkos's existing M_PI-style accessors and reads shorter at the call site;
//     they cannot be constexpr template variables (Kokkos::numbers style) because
//     each is built at runtime from IEEE-754 bit patterns, not a literal.
//   * The former bit-pattern constructor became the static factory
//     DoubleDouble::from_bits(hi, lo): it is namespaced to the type,
//     discoverable, and needs no free-function symbol.
//   * Every single/double-return math function is additionally re-exposed under
//     namespace Kokkos at the bottom of this header (forwarding overloads,
//     mirroring impl/Kokkos_QuadPrecisionMath.hpp) so Kokkos::exp(dd) works
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
// Forward declarations (struct uses them in operator bodies)
// ============================================================
struct DoubleDouble;
KOKKOS_INLINE_FUNCTION DoubleDouble add(DoubleDouble a, DoubleDouble b);
KOKKOS_INLINE_FUNCTION DoubleDouble subtract(DoubleDouble a, DoubleDouble b);
KOKKOS_INLINE_FUNCTION DoubleDouble multiply(DoubleDouble a, DoubleDouble b);
KOKKOS_INLINE_FUNCTION DoubleDouble divide(DoubleDouble a, DoubleDouble b);
KOKKOS_INLINE_FUNCTION DoubleDouble multiply_scalar(DoubleDouble a, double b);
KOKKOS_INLINE_FUNCTION DoubleDouble divide_scalar(DoubleDouble a, double b);
KOKKOS_INLINE_FUNCTION DoubleDouble negate(DoubleDouble a);
KOKKOS_INLINE_FUNCTION DoubleDouble abs(DoubleDouble a);
KOKKOS_INLINE_FUNCTION DoubleDouble sqrt(DoubleDouble a);
KOKKOS_INLINE_FUNCTION DoubleDouble round_to_nearest_int(DoubleDouble a);
KOKKOS_INLINE_FUNCTION DoubleDouble pow_int(DoubleDouble a, int n);
KOKKOS_INLINE_FUNCTION DoubleDouble exp(DoubleDouble a);
KOKKOS_INLINE_FUNCTION DoubleDouble log(DoubleDouble a);
KOKKOS_INLINE_FUNCTION DoubleDouble pow(DoubleDouble a, DoubleDouble b);
KOKKOS_INLINE_FUNCTION void   sinhcosh(DoubleDouble a, DoubleDouble& x, DoubleDouble& y);
KOKKOS_INLINE_FUNCTION void   sincos(DoubleDouble a, DoubleDouble& x, DoubleDouble& y);
KOKKOS_INLINE_FUNCTION DoubleDouble angle(DoubleDouble x, DoubleDouble y);

// ============================================================
// DoubleDouble struct
// ============================================================
struct DoubleDouble {
    double hi;
    double lo;

    KOKKOS_INLINE_FUNCTION DoubleDouble() : hi(0.0), lo(0.0) {}
    KOKKOS_INLINE_FUNCTION DoubleDouble(double h) : hi(h), lo(0.0) {}
    KOKKOS_INLINE_FUNCTION DoubleDouble(double h, double l) : hi(h), lo(l) {}
    KOKKOS_INLINE_FUNCTION DoubleDouble(const DoubleDouble& o) : hi(o.hi), lo(o.lo) {}
    KOKKOS_INLINE_FUNCTION DoubleDouble& operator=(const DoubleDouble& o) { hi=o.hi; lo=o.lo; return *this; }

    // Factory: build a DoubleDouble from the IEEE-754 bit patterns of its two
    // components. Safe on host (memcpy) and device (__longlong_as_double).
    // Replaces the former free bit-pattern constructor function.
    static KOKKOS_INLINE_FUNCTION DoubleDouble from_bits(uint64_t hi_bits, uint64_t lo_bits) {
        double h, l;
#ifndef __CUDA_ARCH__
        std::memcpy(&h, &hi_bits, sizeof(double));
        std::memcpy(&l, &lo_bits, sizeof(double));
#else
        h = __longlong_as_double(static_cast<long long>(hi_bits));
        l = __longlong_as_double(static_cast<long long>(lo_bits));
#endif
        return DoubleDouble(h, l);
    }

    KOKKOS_INLINE_FUNCTION DoubleDouble operator-() const { return negate(*this); }
    KOKKOS_INLINE_FUNCTION DoubleDouble operator+(DoubleDouble b) const { return add(*this, b); }
    KOKKOS_INLINE_FUNCTION DoubleDouble operator-(DoubleDouble b) const { return subtract(*this, b); }
    KOKKOS_INLINE_FUNCTION DoubleDouble operator*(DoubleDouble b) const { return multiply(*this, b); }
    KOKKOS_INLINE_FUNCTION DoubleDouble operator/(DoubleDouble b) const { return divide(*this, b); }
    KOKKOS_INLINE_FUNCTION DoubleDouble operator*(double b)  const { return multiply_scalar(*this, b); }
    KOKKOS_INLINE_FUNCTION DoubleDouble operator/(double b)  const { return divide_scalar(*this, b); }
    KOKKOS_INLINE_FUNCTION DoubleDouble operator+(double b)  const { return add(*this, DoubleDouble(b)); }
    KOKKOS_INLINE_FUNCTION DoubleDouble operator-(double b)  const { return subtract(*this, DoubleDouble(b)); }

    KOKKOS_INLINE_FUNCTION DoubleDouble& operator+=(DoubleDouble b) { *this = *this + b; return *this; }
    KOKKOS_INLINE_FUNCTION DoubleDouble& operator-=(DoubleDouble b) { *this = *this - b; return *this; }
    KOKKOS_INLINE_FUNCTION DoubleDouble& operator*=(DoubleDouble b) { *this = *this * b; return *this; }
    KOKKOS_INLINE_FUNCTION DoubleDouble& operator/=(DoubleDouble b) { *this = *this / b; return *this; }
    KOKKOS_INLINE_FUNCTION DoubleDouble& operator+=(double b) { *this = *this + b; return *this; }
    KOKKOS_INLINE_FUNCTION DoubleDouble& operator-=(double b) { *this = *this - b; return *this; }
    KOKKOS_INLINE_FUNCTION DoubleDouble& operator*=(double b) { *this = multiply_scalar(*this, b); return *this; }
    KOKKOS_INLINE_FUNCTION DoubleDouble& operator/=(double b) { *this = divide_scalar(*this, b); return *this; }

    KOKKOS_INLINE_FUNCTION bool operator==(DoubleDouble b) const { return hi==b.hi && lo==b.lo; }
    KOKKOS_INLINE_FUNCTION bool operator!=(DoubleDouble b) const { return !(*this == b); }
    KOKKOS_INLINE_FUNCTION bool operator<(DoubleDouble b)  const { return hi<b.hi || (hi==b.hi && lo<b.lo); }
    KOKKOS_INLINE_FUNCTION bool operator>(DoubleDouble b)  const { return hi>b.hi || (hi==b.hi && lo>b.lo); }
    KOKKOS_INLINE_FUNCTION bool operator<=(DoubleDouble b) const { return !(b < *this); }
    KOKKOS_INLINE_FUNCTION bool operator>=(DoubleDouble b) const { return !(*this < b); }
};

KOKKOS_INLINE_FUNCTION DoubleDouble operator+(double a, DoubleDouble b) { return add(DoubleDouble(a), b); }
KOKKOS_INLINE_FUNCTION DoubleDouble operator-(double a, DoubleDouble b) { return subtract(DoubleDouble(a), b); }
KOKKOS_INLINE_FUNCTION DoubleDouble operator*(double a, DoubleDouble b) { return multiply_scalar(b, a); }
KOKKOS_INLINE_FUNCTION DoubleDouble operator/(double a, DoubleDouble b) { return divide(DoubleDouble(a), b); }

#ifndef __CUDA_ARCH__
inline std::ostream& operator<<(std::ostream& os, const DoubleDouble& d) {
    os << "[" << std::setprecision(16) << std::scientific << d.hi
       << ", " << d.lo << "]";
    return os;
}
#endif

// ============================================================
// Constants via bit-pattern construction (safe on host + device)
// ============================================================
KOKKOS_INLINE_FUNCTION DoubleDouble DoubleDouble_pi()          { return DoubleDouble::from_bits(0x400921fb54442d18ULL, 0x3ca1a62633145c07ULL); }
KOKKOS_INLINE_FUNCTION DoubleDouble DoubleDouble_e()           { return DoubleDouble::from_bits(0x4005bf0a8b145769ULL, 0x3ca4d57ee2b1013aULL); }
KOKKOS_INLINE_FUNCTION DoubleDouble DoubleDouble_log2()        { return DoubleDouble::from_bits(0x3fe62e42fefa39efULL, 0x3c7abc9e3b39803fULL); }
KOKKOS_INLINE_FUNCTION DoubleDouble DoubleDouble_log10()       { return DoubleDouble::from_bits(0x40026bb1bbb55516ULL, 0xbcaf48ad494ea3eaULL); } // ln(10)
KOKKOS_INLINE_FUNCTION DoubleDouble DoubleDouble_sqrt2()       { return DoubleDouble::from_bits(0x3ff6a09e667f3bcdULL, 0xbc9bdd3413b26456ULL); }
KOKKOS_INLINE_FUNCTION DoubleDouble DoubleDouble_euler_gamma() { return DoubleDouble::from_bits(0x3fe2788cfc6fb619ULL, 0xbc56cb90701fbfabULL); }

// ============================================================
// Primitive arithmetic
// ============================================================

KOKKOS_INLINE_FUNCTION DoubleDouble negate(DoubleDouble a) {
    return DoubleDouble(-a.hi, -a.lo);
}

// TwoSum (Knuth)
KOKKOS_INLINE_FUNCTION DoubleDouble add(DoubleDouble a, DoubleDouble b) {
    double t1 = a.hi + b.hi;
    double e  = t1 - a.hi;
    double t2 = ((b.hi - e) + (a.hi - (t1 - e))) + a.lo + b.lo;
    double hi = t1 + t2;
    double lo = t2 - (hi - t1);
    return DoubleDouble(hi, lo);
}

KOKKOS_INLINE_FUNCTION DoubleDouble subtract(DoubleDouble a, DoubleDouble b) {
    double t1 = a.hi - b.hi;
    double e  = t1 - a.hi;
    double t2 = ((-b.hi - e) + (a.hi - (t1 - e))) + a.lo - b.lo;
    double hi = t1 + t2;
    double lo = t2 - (hi - t1);
    return DoubleDouble(hi, lo);
}

// TwoProduct (Dekker splitting)
KOKKOS_INLINE_FUNCTION DoubleDouble multiply(DoubleDouble a, DoubleDouble b) {
    const double split = 134217729.0;
    double cona = a.hi * split, conb = b.hi * split;
    double a1 = cona - (cona - a.hi), b1 = conb - (conb - b.hi);
    double a2 = a.hi - a1,           b2 = b.hi - b1;
    double c11 = a.hi * b.hi;
    double c21 = (((a1*b1 - c11) + a1*b2) + a2*b1) + a2*b2;
    double c2  = a.hi * b.lo + a.lo * b.hi;
    double t1  = c11 + c2;
    double e   = t1 - c11;
    double t2  = ((c2 - e) + (c11 - (t1 - e))) + c21 + a.lo * b.lo;
    double hi  = t1 + t2;
    double lo  = t2 - (hi - t1);
    return DoubleDouble(hi, lo);
}

KOKKOS_INLINE_FUNCTION DoubleDouble divide(DoubleDouble a, DoubleDouble b) {
    const double split = 134217729.0;
    double s1  = a.hi / b.hi;
    double cona = s1 * split, conb = b.hi * split;
    double a1  = cona - (cona - s1), b1 = conb - (conb - b.hi);
    double a2  = s1 - a1,            b2 = b.hi - b1;
    double c11 = s1 * b.hi;
    double c21 = (((a1*b1 - c11) + a1*b2) + a2*b1) + a2*b2;
    double c2  = s1 * b.lo;
    double t1  = c11 + c2;
    double e   = t1 - c11;
    double t2  = ((c2 - e) + (c11 - (t1 - e))) + c21;
    double t12 = t1 + t2;
    double t22 = t2 - (t12 - t1);
    double t11 = a.hi - t12;
    e = t11 - a.hi;
    double t21 = ((-t12 - e) + (a.hi - (t11 - e))) + a.lo - t22;
    double s2  = (t11 + t21) / b.hi;
    double hi  = s1 + s2;
    double lo  = s2 - (hi - s1);
    return DoubleDouble(hi, lo);
}

KOKKOS_INLINE_FUNCTION DoubleDouble multiply_scalar(DoubleDouble a, double b) {
    const double split = 134217729.0;
    double cona = a.hi * split, conb = b * split;
    double a1   = cona - (cona - a.hi), b1 = conb - (conb - b);
    double a2   = a.hi - a1,            b2 = b - b1;
    double c11  = a.hi * b;
    double c21  = (((a1*b1 - c11) + a1*b2) + a2*b1) + a2*b2;
    double c2   = a.lo * b;
    double t1   = c11 + c2;
    double e    = t1 - c11;
    double t2   = ((c2 - e) + (c11 - (t1 - e))) + c21;
    double hi   = t1 + t2;
    double lo   = t2 - (hi - t1);
    return DoubleDouble(hi, lo);
}

KOKKOS_INLINE_FUNCTION DoubleDouble divide_scalar(DoubleDouble a, double b) {
    const double split = 134217729.0;
    double t1  = a.hi / b;
    double cona = t1 * split, conb = b * split;
    double a1  = cona - (cona - t1), b1 = conb - (conb - b);
    double a2  = t1 - a1,            b2 = b - b1;
    double t12 = t1 * b;
    double t22 = (((a1*b1 - t12) + a1*b2) + a2*b1) + a2*b2;
    double t11 = a.hi - t12;
    double e   = t11 - a.hi;
    double t21 = ((-t12 - e) + (a.hi - (t11 - e))) + a.lo - t22;
    double t2  = (t11 + t21) / b;
    double hi  = t1 + t2;
    double lo  = t2 - (hi - t1);
    return DoubleDouble(hi, lo);
}

// Exact product of two doubles
KOKKOS_INLINE_FUNCTION DoubleDouble two_prod(double da, double db) {
    const double split = 134217729.0;
    double cona = da * split, conb = db * split;
    double a1   = cona - (cona - da), b1 = conb - (conb - db);
    double a2   = da - a1,            b2 = db - b1;
    double s1   = da * db;
    double s2   = (((a1*b1 - s1) + a1*b2) + a2*b1) + a2*b2;
    return DoubleDouble(s1, s2);
}

// ============================================================
// Basic math
// ============================================================

KOKKOS_INLINE_FUNCTION DoubleDouble abs(DoubleDouble a) {
    return (a.hi >= 0.0) ? a : DoubleDouble(-a.hi, -a.lo);
}

// Nearest integer
KOKKOS_INLINE_FUNCTION DoubleDouble round_to_nearest_int(DoubleDouble a) {
    if (a.hi == 0.0) return DoubleDouble(0.0);
    const double T105 = ldexp(1.0, 105); // 2^105
    const double T52  = ldexp(1.0, 52);  // 2^52
    DoubleDouble CON = DoubleDouble(T105, T52);
    if (a.hi >= T105) {
        Kokkos::printf("DDNINT: argument too large\n");
        return DoubleDouble(0.0);
    }
    if (a.hi > 0.0) return subtract(add(a, CON), CON);
    else            return add(subtract(a, CON), CON);
}

KOKKOS_INLINE_FUNCTION DoubleDouble sqrt(DoubleDouble a) {
    if (a.hi == 0.0) return DoubleDouble(0.0);
    if (a.hi < 0.0) {
        Kokkos::printf("DDSQRT: negative argument\n");
        return DoubleDouble(0.0);
    }
    double t1 = 1.0 / Kokkos::sqrt(a.hi);
    double t2 = a.hi * t1;
    DoubleDouble s0 = two_prod(t2, t2);
    DoubleDouble s1 = subtract(a, s0);
    double t3  = 0.5 * s1.hi * t1;
    return add(DoubleDouble(t2), DoubleDouble(t3));
}

// Integer power
KOKKOS_INLINE_FUNCTION DoubleDouble pow_int(DoubleDouble a, int n) {
    const double cl2 = 1.4426950408889633;
    if (a.hi == 0.0) {
        if (n >= 0) return DoubleDouble(0.0);
        Kokkos::printf("DDNPWR: zero base with negative exponent\n");
        return DoubleDouble(0.0);
    }
    int nn = (n < 0) ? -n : n;
    if (nn == 0) return DoubleDouble(1.0);
    if (nn == 1) return (n > 0) ? a : divide(DoubleDouble(1.0), a);
    if (nn == 2) { DoubleDouble r = multiply(a,a); return (n>0) ? r : divide(DoubleDouble(1.0),r); }
    int mn = (int)(cl2 * Kokkos::log((double)nn) + 1.0 + 1.0e-14);
    DoubleDouble s0 = a, s2 = DoubleDouble(1.0);
    int kn = nn;
    for (int j = 1; j <= mn; ++j) {
        int kk = kn / 2;
        if (kn != 2*kk) s2 = multiply(s2, s0);
        kn = kk;
        if (j < mn) s0 = multiply(s0, s0);
    }
    if (n < 0) s2 = divide(DoubleDouble(1.0), s2);
    return s2;
}

// ============================================================
// Exp / Log family
// ============================================================

KOKKOS_INLINE_FUNCTION DoubleDouble exp(DoubleDouble a) {
    const int nq = 6;
    const double eps = 1.0e-32;
    DoubleDouble al2 = DoubleDouble_log2();
    if (a.hi >= 300.0) {
        Kokkos::printf("DDEXP: argument too large\n");
        return DoubleDouble(0.0);
    }
    if (a.hi <= -300.0) return DoubleDouble(0.0);

    DoubleDouble s0 = divide(a, al2);
    DoubleDouble s1 = round_to_nearest_int(s0);
    double t1  = s1.hi;
    int nz     = (int)(t1 + Kokkos::copysign(1.0e-14, t1));
    s0 = subtract(a, multiply(al2, s1));

    if (s0.hi == 0.0) {
        return DoubleDouble(ldexp(1.0, nz)); // result = 2^nz exactly
    }
    // Scale down by 2^nq then square nq times
    s1 = multiply_scalar(s0, ldexp(1.0, -nq));
    DoubleDouble s2 = DoubleDouble(1.0), s3 = DoubleDouble(1.0);
    for (int l1 = 1; l1 <= 100; ++l1) {
        s0 = multiply(s2, s1);
        s2 = divide_scalar(s0, (double)l1);
        s0 = add(s3, s2);
        s3 = s0;
        if (Kokkos::fabs(s2.hi) <= eps * Kokkos::fabs(s3.hi)) break;
        if (l1 == 100) { Kokkos::printf("DDEXP: iteration limit\n"); return DoubleDouble(0.0); }
    }
    for (int i = 0; i < nq; ++i) s3 = multiply(s3, s3);

    return multiply_scalar(s3, ldexp(1.0, nz)); // multiply by 2^nz
}

KOKKOS_INLINE_FUNCTION DoubleDouble log(DoubleDouble a) {
    if (a.hi <= 0.0) {
        Kokkos::printf("DDLOG: non-positive argument\n");
        return DoubleDouble(0.0);
    }
    // Initial approximation then 3 Newton steps: b <- b + (a - exp(b)) / exp(b)
    DoubleDouble b = DoubleDouble(Kokkos::log(a.hi));
    for (int k = 0; k < 3; ++k) {
        DoubleDouble s0 = exp(b);
        DoubleDouble s1 = subtract(a, s0);
        DoubleDouble s2 = divide(s1, s0);
        b = add(b, s2);
    }
    return b;
}

KOKKOS_INLINE_FUNCTION DoubleDouble log2(DoubleDouble a) {
    return divide(log(a), DoubleDouble_log2());
}

KOKKOS_INLINE_FUNCTION DoubleDouble log10(DoubleDouble a) {
    return divide(log(a), DoubleDouble_log10());
}

KOKKOS_INLINE_FUNCTION DoubleDouble log1p(DoubleDouble a) {
    // log(1+a); use direct formula for moderate a
    return log(add(DoubleDouble(1.0), a));
}

KOKKOS_INLINE_FUNCTION DoubleDouble exp2(DoubleDouble a) {
    return exp(multiply(a, DoubleDouble_log2()));
}

KOKKOS_INLINE_FUNCTION DoubleDouble exp10(DoubleDouble a) {
    return exp(multiply(a, DoubleDouble_log10()));
}

KOKKOS_INLINE_FUNCTION DoubleDouble expm1(DoubleDouble a) {
    if (Kokkos::fabs(a.hi) > 0.5) {
        // |exp(a)-1| > e^0.5-1 ~ 0.65: subtraction of 1 causes no significant cancellation
        return subtract(exp(a), DoubleDouble(1.0));
    }
    // Taylor series: a + a²/2! + a³/3! + ...
    // Avoids catastrophic cancellation of exp(a)-1 near a=0
    DoubleDouble sum = a, term = a;
    for (int k = 2; k <= 50; ++k) {
        term = divide_scalar(multiply(term, a), (double)k);
        sum  = add(sum, term);
        if (Kokkos::fabs(term.hi) < 1.0e-32 * Kokkos::fabs(sum.hi)) break;
    }
    return sum;
}

// ============================================================
// Trig — internal combined cos+sin, then derived
// ============================================================

// sincos: compute (cos(a), sin(a)) via argument reduction + Taylor series
// x = cos(a), y = sin(a)
KOKKOS_INLINE_FUNCTION void sincos(DoubleDouble a, DoubleDouble& x, DoubleDouble& y) {
    const int itrmx = 1000, nq = 5;
    const double eps = 1.0e-32;
    if (a.hi == 0.0) { x = DoubleDouble(1.0); y = DoubleDouble(0.0); return; }
    if (a.hi >= 1.0e60) {
        Kokkos::printf("DDCSSNR: argument too large\n");
        x = DoubleDouble(0.0); y = DoubleDouble(0.0); return;
    }
    DoubleDouble pi2 = multiply_scalar(DoubleDouble_pi(), 2.0);
    DoubleDouble s1  = divide(a, pi2);
    DoubleDouble s2  = round_to_nearest_int(s1);
    DoubleDouble s3  = subtract(a, multiply(pi2, s2));
    if (s3.hi == 0.0) { x = DoubleDouble(1.0); y = DoubleDouble(0.0); return; }
    int is = (s3.hi < 0.0) ? -1 : 1;
    double scale = 1.0 / (double)(1 << nq);
    DoubleDouble s0 = multiply_scalar(s3, scale);
    DoubleDouble s1t = s0;
    DoubleDouble s2sq = multiply(s0, s0);
    // Sine series: s0 accumulates sin(s3/2^nq)
    for (int i1 = 1; i1 <= itrmx; ++i1) {
        double t2 = -(2.0*i1) * (2.0*i1 + 1.0);
        DoubleDouble s3t = multiply(s2sq, s1t);
        s1t = divide_scalar(s3t, t2);
        s3t = add(s1t, s0);
        s0  = s3t;
        if (Kokkos::fabs(s1t.hi) < eps) break;
        if (i1 == itrmx) { Kokkos::printf("DDCSSNR: iteration limit\n"); return; }
    }
    // Double-angle nq times: sin(2x) = 2*sin(x)*cos(x), cos(2x) = 1 - 2*sin²(x)
    DoubleDouble f2 = DoubleDouble(0.5);
    DoubleDouble s4 = multiply(s0, s0);
    DoubleDouble s5 = subtract(f2, s4);
    s0 = multiply_scalar(s5, 2.0);
    for (int j = 2; j <= nq; ++j) {
        s4 = multiply(s0, s0);
        s5 = subtract(s4, f2);
        s0 = multiply_scalar(s5, 2.0);
    }
    // s0 now = cos(s3). Recover sin.
    s4 = multiply(s0, s0);
    s5 = subtract(DoubleDouble(1.0), s4);
    s1t = sqrt(s5);
    if (is < 0) { s1t.hi = -s1t.hi; s1t.lo = -s1t.lo; }
    x = s0; y = s1t;
}

KOKKOS_INLINE_FUNCTION DoubleDouble sin(DoubleDouble a) {
    DoubleDouble c, s; sincos(a, c, s); return s;
}
KOKKOS_INLINE_FUNCTION DoubleDouble cos(DoubleDouble a) {
    DoubleDouble c, s; sincos(a, c, s); return c;
}
KOKKOS_INLINE_FUNCTION DoubleDouble tan(DoubleDouble a) {
    DoubleDouble c, s; sincos(a, c, s); return divide(s, c);
}

// Angle of point (x, y) = atan2(y, x). Internal DDFUN primitive (DDANG); the
// public STL-ordered wrapper is atan2(y, x) below.
KOKKOS_INLINE_FUNCTION DoubleDouble angle(DoubleDouble x, DoubleDouble y) {
    DoubleDouble pi = DoubleDouble_pi();
    if (x.hi == 0.0 && y.hi == 0.0) return DoubleDouble(0.0);
    if (x.hi == 0.0) return (y.hi > 0.0) ? multiply_scalar(pi, 0.5) : multiply_scalar(pi, -0.5);
    if (y.hi == 0.0) return (x.hi > 0.0) ? DoubleDouble(0.0) : pi;
    // Normalize
    DoubleDouble r = sqrt(add(multiply(x,x), multiply(y,y)));
    DoubleDouble nx = divide(x, r), ny = divide(y, r);
    // Initial approximation
    DoubleDouble a = DoubleDouble(Kokkos::atan2(ny.hi, nx.hi));
    bool use_x = (Kokkos::fabs(nx.hi) <= Kokkos::fabs(ny.hi));
    DoubleDouble target = use_x ? nx : ny;
    for (int k = 0; k < 3; ++k) {
        DoubleDouble sin_a, cos_a;
        sincos(a, cos_a, sin_a);
        DoubleDouble corr;
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

KOKKOS_INLINE_FUNCTION DoubleDouble asin(DoubleDouble a) {
    if (Kokkos::fabs(a.hi) > 1.0) {
        Kokkos::printf("DDASIN: argument out of range\n");
        return DoubleDouble(0.0);
    }
    DoubleDouble t = sqrt(subtract(DoubleDouble(1.0), multiply(a, a)));
    return angle(t, a); // atan2(a, sqrt(1-a^2))
}
KOKKOS_INLINE_FUNCTION DoubleDouble acos(DoubleDouble a) {
    if (Kokkos::fabs(a.hi) > 1.0) {
        Kokkos::printf("DDACOS: argument out of range\n");
        return DoubleDouble(0.0);
    }
    DoubleDouble t = sqrt(subtract(DoubleDouble(1.0), multiply(a, a)));
    return angle(a, t); // atan2(sqrt(1-a^2), a)
}
KOKKOS_INLINE_FUNCTION DoubleDouble atan(DoubleDouble a) {
    return angle(DoubleDouble(1.0), a); // atan2(a, 1)
}
KOKKOS_INLINE_FUNCTION DoubleDouble atan2(DoubleDouble y, DoubleDouble x) {
    return angle(x, y);
}

// ============================================================
// Hyperbolic — internal combined cosh+sinh, then derived
// ============================================================

// x = cosh(a), y = sinh(a)
KOKKOS_INLINE_FUNCTION void sinhcosh(DoubleDouble a, DoubleDouble& x, DoubleDouble& y) {
    DoubleDouble s0 = exp(a);
    DoubleDouble s1 = divide(DoubleDouble(1.0), s0);
    x = multiply_scalar(add(s0, s1), 0.5);
    y = multiply_scalar(subtract(s0, s1), 0.5);
}

KOKKOS_INLINE_FUNCTION DoubleDouble sinh(DoubleDouble a) {
    DoubleDouble c, s; sinhcosh(a, c, s); return s;
}
KOKKOS_INLINE_FUNCTION DoubleDouble cosh(DoubleDouble a) {
    DoubleDouble c, s; sinhcosh(a, c, s); return c;
}
KOKKOS_INLINE_FUNCTION DoubleDouble tanh(DoubleDouble a) {
    // tanh(x) = expm1(2x) / (expm1(2x) + 2), reflected for negative x
    // Avoids dividing two nearly-equal large numbers from sinhcosh
    if (a.hi < 0.0) return negate(tanh(negate(a)));
    DoubleDouble e = expm1(multiply_scalar(a, 2.0));
    return divide(e, add(e, DoubleDouble(2.0)));
}

KOKKOS_INLINE_FUNCTION DoubleDouble asinh(DoubleDouble a) {
    // Reflect: asinh(-a) = -asinh(a). For positive a, a + sqrt(a²+1) >= 1 always,
    // so log argument never causes cancellation.
    if (a.hi < 0.0) return negate(asinh(negate(a)));
    return log(add(a, sqrt(add(multiply(a, a), DoubleDouble(1.0)))));
}
KOKKOS_INLINE_FUNCTION DoubleDouble acosh(DoubleDouble a) {
    if (a.hi < 1.0) { Kokkos::printf("DDACOSH: argument < 1\n"); return DoubleDouble(0.0); }
    DoubleDouble t1 = subtract(multiply(a, a), DoubleDouble(1.0));
    return log(add(a, sqrt(t1)));
}
KOKKOS_INLINE_FUNCTION DoubleDouble atanh(DoubleDouble a) {
    if (Kokkos::fabs(a.hi) >= 1.0) { Kokkos::printf("DDATANH: |argument| >= 1\n"); return DoubleDouble(0.0); }
    DoubleDouble t1 = add(DoubleDouble(1.0), a);
    DoubleDouble t2 = subtract(DoubleDouble(1.0), a);
    return multiply_scalar(log(divide(t1, t2)), 0.5);
}

// ============================================================
// Multi-argument operations
// ============================================================

KOKKOS_INLINE_FUNCTION DoubleDouble pow(DoubleDouble a, DoubleDouble b) {
    if (a.hi <= 0.0) {
        if (a.hi == 0.0 && b.hi > 0.0) return DoubleDouble(0.0);
        Kokkos::printf("DDPOW: non-positive base\n");
        return DoubleDouble(0.0);
    }
    return exp(multiply(log(a), b));
}

KOKKOS_INLINE_FUNCTION DoubleDouble hypot(DoubleDouble a, DoubleDouble b) {
    return sqrt(add(multiply(a, a), multiply(b, b)));
}

KOKKOS_INLINE_FUNCTION DoubleDouble ceil(DoubleDouble a);
KOKKOS_INLINE_FUNCTION DoubleDouble floor(DoubleDouble a);
KOKKOS_INLINE_FUNCTION DoubleDouble trunc(DoubleDouble a);
KOKKOS_INLINE_FUNCTION DoubleDouble round(DoubleDouble a);

KOKKOS_INLINE_FUNCTION DoubleDouble fmod(DoubleDouble a, DoubleDouble b) {
    DoubleDouble q = divide(a, b);
    DoubleDouble qt = trunc(q);
    return subtract(a, multiply(b, qt));
}

KOKKOS_INLINE_FUNCTION DoubleDouble remainder(DoubleDouble a, DoubleDouble b) {
    DoubleDouble q = divide(a, b);
    DoubleDouble qn = round_to_nearest_int(q);
    return subtract(a, multiply(b, qn));
}

KOKKOS_INLINE_FUNCTION DoubleDouble copysign(DoubleDouble a, DoubleDouble b) {
    DoubleDouble r = abs(a);
    if (b.hi < 0.0 || (b.hi == 0.0 && b.lo < 0.0)) return negate(r);
    return r;
}

KOKKOS_INLINE_FUNCTION DoubleDouble fmax(DoubleDouble a, DoubleDouble b) {
    return (a > b) ? a : b;
}
KOKKOS_INLINE_FUNCTION DoubleDouble fmin(DoubleDouble a, DoubleDouble b) {
    return (a < b) ? a : b;
}
KOKKOS_INLINE_FUNCTION DoubleDouble fdim(DoubleDouble a, DoubleDouble b) {
    return (a > b) ? subtract(a, b) : DoubleDouble(0.0);
}
KOKKOS_INLINE_FUNCTION DoubleDouble fma(DoubleDouble a, DoubleDouble b, DoubleDouble c) {
    return add(multiply(a, b), c);
}

// ============================================================
// Rounding
// ============================================================

KOKKOS_INLINE_FUNCTION DoubleDouble floor(DoubleDouble a) {
    DoubleDouble n = round_to_nearest_int(a);
    if (n > a) return subtract(n, DoubleDouble(1.0));
    return n;
}
KOKKOS_INLINE_FUNCTION DoubleDouble ceil(DoubleDouble a) {
    DoubleDouble n = round_to_nearest_int(a);
    if (n < a) return add(n, DoubleDouble(1.0));
    return n;
}
KOKKOS_INLINE_FUNCTION DoubleDouble trunc(DoubleDouble a) {
    return (a.hi >= 0.0) ? floor(a) : ceil(a);
}
KOKKOS_INLINE_FUNCTION DoubleDouble round(DoubleDouble a) {
    return round_to_nearest_int(a);
}

// ============================================================
// Special functions (in header, not benchmarked)
// ============================================================

// erf — Taylor series for |z| < ~9, asymptotic otherwise
KOKKOS_INLINE_FUNCTION DoubleDouble erf(DoubleDouble z) {
    const double eps = 1.0e-32;
    if (z.hi == 0.0) return DoubleDouble(0.0);
    // threshold: sqrt(104 * ln2) ≈ 8.48
    const double large = 8.5;
    if (z.hi >  large) return DoubleDouble( 1.0);
    if (z.hi < -large) return DoubleDouble(-1.0);

    DoubleDouble z2 = multiply(z, z);
    int sign = (z.hi >= 0.0) ? 1 : -1;
    DoubleDouble az = abs(z);

    if (Kokkos::fabs(z.hi) < 9.0) {
        // Series: erf(z) = (2/sqrt(pi)) * exp(-z^2) * sum_k 2^k * z^{2k+1} / (1*3*...*(2k+1))
        DoubleDouble t1 = DoubleDouble(0.0), t2 = az, t3 = DoubleDouble(1.0);
        for (int k = 0; k <= 100; ++k) {
            if (k > 0) {
                t2 = multiply_scalar(multiply(z2, t2), 2.0);
                t3 = multiply_scalar(t3, 2.0*k + 1.0);
            }
            DoubleDouble t4 = divide(t2, t3);
            DoubleDouble t1new = add(t1, t4);
            if (Kokkos::fabs(t4.hi) < eps * Kokkos::fabs(t1new.hi)) { t1 = t1new; break; }
            t1 = t1new;
        }
        DoubleDouble result = multiply_scalar(divide(multiply_scalar(t1, 2.0),
                                multiply(sqrt(DoubleDouble_pi()), exp(z2))), 1.0);
        return (sign > 0) ? result : negate(result);
    } else {
        // Asymptotic: erf(z) = 1 - erfc(z)
        // erfc(z) = exp(-z^2)/sqrt(pi) * sum_k (-1)^k * (2k-1)!! / (2^k * z^{2k+1})
        DoubleDouble t1 = DoubleDouble(0.0), t2 = DoubleDouble(1.0), t3 = az;
        for (int k = 0; k <= 100; ++k) {
            if (k > 0) {
                t2 = multiply_scalar(t2, -(2.0*k - 1.0));
                t3 = multiply(t3, multiply_scalar(z2, 2.0));
            }
            DoubleDouble t4 = divide(t2, t3);
            DoubleDouble t1new = add(t1, t4);
            if (Kokkos::fabs(divide(t4, t1new).hi) < eps) { t1 = t1new; break; }
            t1 = t1new;
        }
        DoubleDouble erfc_val = divide(t1, multiply(sqrt(DoubleDouble_pi()), exp(z2)));
        DoubleDouble erf_val  = subtract(DoubleDouble(1.0), erfc_val);
        return (sign > 0) ? erf_val : negate(erf_val);
    }
}

KOKKOS_INLINE_FUNCTION DoubleDouble erfc(DoubleDouble z) {
    return subtract(DoubleDouble(1.0), erf(z));
}

// gamma — Lanczos approximation at DD precision
KOKKOS_INLINE_FUNCTION DoubleDouble tgamma(DoubleDouble a) {
    if (a.hi < 0.5) {
        DoubleDouble pi = DoubleDouble_pi();
        DoubleDouble sin_pi_a = sin(multiply(pi, a));
        return divide(pi, multiply(sin_pi_a, tgamma(subtract(DoubleDouble(1.0), a))));
    }
    // Lanczos g=7 coefficients (no static — not device-safe)
    const double c0 =  0.99999999999980993;
    const double c1 =  676.5203681218851;
    const double c2 = -1259.1392167224028;
    const double c3 =  771.32342877765313;
    const double c4 = -176.61502916214059;
    const double c5 =  12.507343278686905;
    const double c6 = -0.13857109526572012;
    const double c7 =  9.9843695780195716e-6;
    const double c8 =  1.5056327351493116e-7;
    DoubleDouble x = subtract(a, DoubleDouble(1.0));
    DoubleDouble t = add(x, DoubleDouble(7.5));
    DoubleDouble s = DoubleDouble(c0);
    s = add(s, divide(DoubleDouble(c1), add(x, DoubleDouble(1.0))));
    s = add(s, divide(DoubleDouble(c2), add(x, DoubleDouble(2.0))));
    s = add(s, divide(DoubleDouble(c3), add(x, DoubleDouble(3.0))));
    s = add(s, divide(DoubleDouble(c4), add(x, DoubleDouble(4.0))));
    s = add(s, divide(DoubleDouble(c5), add(x, DoubleDouble(5.0))));
    s = add(s, divide(DoubleDouble(c6), add(x, DoubleDouble(6.0))));
    s = add(s, divide(DoubleDouble(c7), add(x, DoubleDouble(7.0))));
    s = add(s, divide(DoubleDouble(c8), add(x, DoubleDouble(8.0))));
    DoubleDouble two_pi_sqrt = DoubleDouble(2.5066282746310002); // sqrt(2*pi)
    return multiply(multiply(two_pi_sqrt, s),
                 multiply(pow(t, add(x, DoubleDouble(0.5))), exp(negate(t))));
}

// Bessel J0 via series
KOKKOS_INLINE_FUNCTION DoubleDouble bessel_j0(DoubleDouble x) {
    const double eps = 1.0e-32;
    DoubleDouble x2 = multiply_scalar(multiply(x, x), -0.25);
    DoubleDouble term = DoubleDouble(1.0), sum = DoubleDouble(1.0);
    for (int k = 1; k <= 100; ++k) {
        term = divide_scalar(multiply(term, x2), (double)(k*k));
        sum  = add(sum, term);
        if (Kokkos::fabs(term.hi) < eps * Kokkos::fabs(sum.hi)) break;
    }
    return sum;
}

KOKKOS_INLINE_FUNCTION DoubleDouble bessel_j1(DoubleDouble x) {
    const double eps = 1.0e-32;
    DoubleDouble x2 = multiply_scalar(multiply(x, x), -0.25);
    DoubleDouble term = multiply_scalar(x, 0.5), sum = term;
    for (int k = 1; k <= 100; ++k) {
        term = divide_scalar(multiply(term, x2), (double)(k * (k+1)));
        sum  = add(sum, term);
        if (Kokkos::fabs(term.hi) < eps * Kokkos::fabs(sum.hi)) break;
    }
    return sum;
}

KOKKOS_INLINE_FUNCTION DoubleDouble bessel_jn(int n, DoubleDouble x) {
    if (n == 0) return bessel_j0(x);
    if (n == 1) return bessel_j1(x);
    // Downward recurrence
    DoubleDouble j0 = bessel_j0(x), j1 = bessel_j1(x);
    DoubleDouble jm1 = j0, j_cur = j1;
    for (int k = 1; k < n; ++k) {
        DoubleDouble jp1 = subtract(multiply_scalar(divide(j_cur, x), 2.0*k), jm1);
        jm1   = j_cur;
        j_cur = jp1;
    }
    return j_cur;
}

KOKKOS_INLINE_FUNCTION DoubleDouble bessel_y0(DoubleDouble x) {
    // Y0(x) = (2/pi)*(J0(x)*log(x/2) + sum...)  — simplified
    DoubleDouble two_over_pi = divide_scalar(DoubleDouble(2.0), DoubleDouble_pi().hi);
    DoubleDouble j0 = bessel_j0(x);
    return multiply(two_over_pi, multiply(j0, log(multiply_scalar(x, 0.5))));
}
KOKKOS_INLINE_FUNCTION DoubleDouble bessel_y1(DoubleDouble x) {
    DoubleDouble two_over_pi = divide_scalar(DoubleDouble(2.0), DoubleDouble_pi().hi);
    DoubleDouble j1 = bessel_j1(x);
    return multiply(two_over_pi, multiply(j1, log(multiply_scalar(x, 0.5))));
}
KOKKOS_INLINE_FUNCTION DoubleDouble bessel_yn(int n, DoubleDouble x) {
    if (n == 0) return bessel_y0(x);
    if (n == 1) return bessel_y1(x);
    DoubleDouble y0 = bessel_y0(x), y1 = bessel_y1(x);
    DoubleDouble ym1 = y0, y_cur = y1;
    for (int k = 1; k < n; ++k) {
        DoubleDouble yp1 = subtract(multiply_scalar(divide(y_cur, x), 2.0*k), ym1);
        ym1   = y_cur;
        y_cur = yp1;
    }
    return y_cur;
}

// Zeta function — Euler-Maclaurin for s > 1
KOKKOS_INLINE_FUNCTION DoubleDouble zeta(DoubleDouble s) {
    if (s.hi <= 1.0) { Kokkos::printf("DDZETA: s <= 1\n"); return DoubleDouble(0.0); }
    const int N = 50;
    DoubleDouble sum = DoubleDouble(0.0);
    for (int k = 1; k <= N; ++k)
        sum = add(sum, exp(multiply(negate(s), log(DoubleDouble((double)k)))));
    // tail correction integral: N^{1-s}/(s-1)
    DoubleDouble tail = divide(exp(multiply(subtract(DoubleDouble(1.0), s), log(DoubleDouble((double)N)))),
                         subtract(s, DoubleDouble(1.0)));
    return add(sum, tail);
}

// Exponential integral Ei(x) via series (x > 0)
KOKKOS_INLINE_FUNCTION DoubleDouble expint(DoubleDouble x) {
    DoubleDouble eg = DoubleDouble_euler_gamma();
    DoubleDouble sum = add(eg, log(abs(x)));
    DoubleDouble term = x;
    for (int k = 1; k <= 100; ++k) {
        sum = add(sum, divide_scalar(term, (double)(k * k)));
        term = multiply(term, x);
        if (Kokkos::fabs(term.hi) * 1e-32 < Kokkos::fabs(sum.hi)) break;
    }
    return sum;
}

// Incomplete gamma P(a,x) via series
KOKKOS_INLINE_FUNCTION DoubleDouble incgamma(DoubleDouble a, DoubleDouble x) {
    const double eps = 1.0e-32;
    DoubleDouble term = divide(exp(negate(x)), a);
    DoubleDouble sum  = term;
    for (int k = 1; k <= 100; ++k) {
        term = multiply(term, divide(x, add(a, DoubleDouble((double)k))));
        sum  = add(sum, term);
        if (Kokkos::fabs(term.hi) < eps * Kokkos::fabs(sum.hi)) break;
    }
    return multiply(sum, exp(multiply(a, log(x))));
}

} // namespace Experimental
} // namespace Kokkos

// ============================================================
// Re-exposure under namespace Kokkos (T0.4)
// ============================================================
// Mirrors impl/Kokkos_QuadPrecisionMath.hpp's __float128 overloads so user code
// can call Kokkos::exp(dd) identically to Kokkos::exp(double)/Kokkos::exp(
// __float128). One-line forwards to the Kokkos::Experimental implementations.
// NOTE: add/subtract/multiply/divide are deliberately NOT forwarded here — those
// are reached via operators and explicit ADL, not as Kokkos::add etc.
namespace Kokkos {
// clang-format off
KOKKOS_INLINE_FUNCTION Experimental::DoubleDouble abs(Experimental::DoubleDouble x)   { return Experimental::abs(x); }
KOKKOS_INLINE_FUNCTION Experimental::DoubleDouble sqrt(Experimental::DoubleDouble x)  { return Experimental::sqrt(x); }
KOKKOS_INLINE_FUNCTION Experimental::DoubleDouble exp(Experimental::DoubleDouble x)   { return Experimental::exp(x); }
KOKKOS_INLINE_FUNCTION Experimental::DoubleDouble exp2(Experimental::DoubleDouble x)  { return Experimental::exp2(x); }
KOKKOS_INLINE_FUNCTION Experimental::DoubleDouble exp10(Experimental::DoubleDouble x) { return Experimental::exp10(x); }
KOKKOS_INLINE_FUNCTION Experimental::DoubleDouble expm1(Experimental::DoubleDouble x) { return Experimental::expm1(x); }
KOKKOS_INLINE_FUNCTION Experimental::DoubleDouble log(Experimental::DoubleDouble x)   { return Experimental::log(x); }
KOKKOS_INLINE_FUNCTION Experimental::DoubleDouble log2(Experimental::DoubleDouble x)  { return Experimental::log2(x); }
KOKKOS_INLINE_FUNCTION Experimental::DoubleDouble log10(Experimental::DoubleDouble x) { return Experimental::log10(x); }
KOKKOS_INLINE_FUNCTION Experimental::DoubleDouble log1p(Experimental::DoubleDouble x) { return Experimental::log1p(x); }
KOKKOS_INLINE_FUNCTION Experimental::DoubleDouble sin(Experimental::DoubleDouble x)   { return Experimental::sin(x); }
KOKKOS_INLINE_FUNCTION Experimental::DoubleDouble cos(Experimental::DoubleDouble x)   { return Experimental::cos(x); }
KOKKOS_INLINE_FUNCTION Experimental::DoubleDouble tan(Experimental::DoubleDouble x)   { return Experimental::tan(x); }
KOKKOS_INLINE_FUNCTION Experimental::DoubleDouble asin(Experimental::DoubleDouble x)  { return Experimental::asin(x); }
KOKKOS_INLINE_FUNCTION Experimental::DoubleDouble acos(Experimental::DoubleDouble x)  { return Experimental::acos(x); }
KOKKOS_INLINE_FUNCTION Experimental::DoubleDouble atan(Experimental::DoubleDouble x)  { return Experimental::atan(x); }
KOKKOS_INLINE_FUNCTION Experimental::DoubleDouble atan2(Experimental::DoubleDouble y, Experimental::DoubleDouble x) { return Experimental::atan2(y, x); }
KOKKOS_INLINE_FUNCTION Experimental::DoubleDouble sinh(Experimental::DoubleDouble x)  { return Experimental::sinh(x); }
KOKKOS_INLINE_FUNCTION Experimental::DoubleDouble cosh(Experimental::DoubleDouble x)  { return Experimental::cosh(x); }
KOKKOS_INLINE_FUNCTION Experimental::DoubleDouble tanh(Experimental::DoubleDouble x)  { return Experimental::tanh(x); }
KOKKOS_INLINE_FUNCTION Experimental::DoubleDouble asinh(Experimental::DoubleDouble x) { return Experimental::asinh(x); }
KOKKOS_INLINE_FUNCTION Experimental::DoubleDouble acosh(Experimental::DoubleDouble x) { return Experimental::acosh(x); }
KOKKOS_INLINE_FUNCTION Experimental::DoubleDouble atanh(Experimental::DoubleDouble x) { return Experimental::atanh(x); }
KOKKOS_INLINE_FUNCTION Experimental::DoubleDouble pow(Experimental::DoubleDouble a, Experimental::DoubleDouble b) { return Experimental::pow(a, b); }
KOKKOS_INLINE_FUNCTION Experimental::DoubleDouble hypot(Experimental::DoubleDouble a, Experimental::DoubleDouble b) { return Experimental::hypot(a, b); }
KOKKOS_INLINE_FUNCTION Experimental::DoubleDouble fmod(Experimental::DoubleDouble a, Experimental::DoubleDouble b) { return Experimental::fmod(a, b); }
KOKKOS_INLINE_FUNCTION Experimental::DoubleDouble remainder(Experimental::DoubleDouble a, Experimental::DoubleDouble b) { return Experimental::remainder(a, b); }
KOKKOS_INLINE_FUNCTION Experimental::DoubleDouble copysign(Experimental::DoubleDouble a, Experimental::DoubleDouble b) { return Experimental::copysign(a, b); }
KOKKOS_INLINE_FUNCTION Experimental::DoubleDouble fmax(Experimental::DoubleDouble a, Experimental::DoubleDouble b) { return Experimental::fmax(a, b); }
KOKKOS_INLINE_FUNCTION Experimental::DoubleDouble fmin(Experimental::DoubleDouble a, Experimental::DoubleDouble b) { return Experimental::fmin(a, b); }
KOKKOS_INLINE_FUNCTION Experimental::DoubleDouble fdim(Experimental::DoubleDouble a, Experimental::DoubleDouble b) { return Experimental::fdim(a, b); }
KOKKOS_INLINE_FUNCTION Experimental::DoubleDouble fma(Experimental::DoubleDouble a, Experimental::DoubleDouble b, Experimental::DoubleDouble c) { return Experimental::fma(a, b, c); }
KOKKOS_INLINE_FUNCTION Experimental::DoubleDouble ceil(Experimental::DoubleDouble x)  { return Experimental::ceil(x); }
KOKKOS_INLINE_FUNCTION Experimental::DoubleDouble floor(Experimental::DoubleDouble x) { return Experimental::floor(x); }
KOKKOS_INLINE_FUNCTION Experimental::DoubleDouble round(Experimental::DoubleDouble x) { return Experimental::round(x); }
KOKKOS_INLINE_FUNCTION Experimental::DoubleDouble trunc(Experimental::DoubleDouble x) { return Experimental::trunc(x); }
KOKKOS_INLINE_FUNCTION Experimental::DoubleDouble erf(Experimental::DoubleDouble x)   { return Experimental::erf(x); }
KOKKOS_INLINE_FUNCTION Experimental::DoubleDouble erfc(Experimental::DoubleDouble x)  { return Experimental::erfc(x); }
KOKKOS_INLINE_FUNCTION Experimental::DoubleDouble tgamma(Experimental::DoubleDouble x){ return Experimental::tgamma(x); }
// clang-format on
}  // namespace Kokkos
