// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

// NOTE: This header is NOT part of upstream Kokkos. It is a local extension to
// impl/Kokkos_QuadPrecisionMath.hpp that adds __complex128 overloads in
// namespace Kokkos, mirroring that header's __float128 overloads. It exists so a
// host-side quadmath oracle can be expressed as Kokkos::exp((__complex128)z)
// etc., matching the real-oracle plumbing adopted in T0.0. A repo-side copy
// lives at patches/kokkos_complex_quad_math.hpp; see patches/README.md.

#ifndef KOKKOS_COMPLEX_QUAD_PRECISION_MATH_HPP
#define KOKKOS_COMPLEX_QUAD_PRECISION_MATH_HPP

#include <Kokkos_Macros.hpp>

#if defined(KOKKOS_ENABLE_LIBQUADMATH)

#include <impl/Kokkos_QuadPrecisionMath.hpp>

#include <quadmath.h>

#if !(defined(__FLOAT128__) || defined(__SIZEOF_FLOAT128__))
#error __float128 not supported on this host
#endif

//<editor-fold desc="Common mathematical functions __complex128 overloads">
namespace Kokkos {
// clang-format off
// Component access (return __float128)
inline __float128 abs(__complex128 z)  { return ::cabsq(z); }
inline __float128 real(__complex128 z) { return ::crealq(z); }
inline __float128 imag(__complex128 z) { return ::cimagq(z); }
// Conjugate
inline __complex128 conj(__complex128 z) { return ::conjq(z); }
// Exponential functions
inline __complex128 exp(__complex128 z)   { return ::cexpq(z); }
inline __complex128 log(__complex128 z)   { return ::clogq(z); }
inline __complex128 log10(__complex128 z) { return ::clog10q(z); }
// Power functions
inline __complex128 pow(__complex128 x, __complex128 y) { return ::cpowq(x, y); }
inline __complex128 sqrt(__complex128 z) { return ::csqrtq(z); }
// Trigonometric functions
inline __complex128 sin(__complex128 z)  { return ::csinq(z); }
inline __complex128 cos(__complex128 z)  { return ::ccosq(z); }
inline __complex128 tan(__complex128 z)  { return ::ctanq(z); }
inline __complex128 asin(__complex128 z) { return ::casinq(z); }
inline __complex128 acos(__complex128 z) { return ::cacosq(z); }
inline __complex128 atan(__complex128 z) { return ::catanq(z); }
// Hyperbolic functions
inline __complex128 sinh(__complex128 z)  { return ::csinhq(z); }
inline __complex128 cosh(__complex128 z)  { return ::ccoshq(z); }
inline __complex128 tanh(__complex128 z)  { return ::ctanhq(z); }
inline __complex128 asinh(__complex128 z) { return ::casinhq(z); }
inline __complex128 acosh(__complex128 z) { return ::cacoshq(z); }
inline __complex128 atanh(__complex128 z) { return ::catanhq(z); }
// clang-format on
}  // namespace Kokkos
//</editor-fold>

#endif

#endif
