//
// QCDLoop + Kokkos 2025
//
// Authors: Reet Barik      : rbarik@anl.gov
//          Taylor Childers : jchilders@anl.gov
//
// Minimal Quad Complex Type
// Do NOT use Kokkos::complex<fp128_t> - implement minimal custom type instead

#pragma once

#ifdef KOKKOS_ENABLE_CUDA

#include <type_traits>
#include "quad_math.hpp"

namespace ql {

// Minimal quad_complex type for quad precision complex arithmetic
// Only implements what is actually needed by kernels
struct quad_complex {
    fp128_t re;
    fp128_t im;

    KOKKOS_INLINE_FUNCTION
    quad_complex() = default;

    KOKKOS_INLINE_FUNCTION
    quad_complex(fp128_t r, fp128_t i = 0)
        : re(r), im(i) {}
    
    // Explicit copy constructor - provides NVCC with an actual function body
    // to emit for device code (= default fails to synthesize device-side symbols)
    KOKKOS_INLINE_FUNCTION
    quad_complex(const quad_complex& other) : re(other.re), im(other.im) {}
    
    // Explicit copy assignment operator - same rationale as copy constructor
    KOKKOS_INLINE_FUNCTION
    quad_complex& operator=(const quad_complex& other) { re = other.re; im = other.im; return *this; }
    
    // Destructor - use default to maintain trivial copyability
    KOKKOS_INLINE_FUNCTION
    ~quad_complex() = default;
    
    KOKKOS_INLINE_FUNCTION
    fp128_t real() const { return re; }

    KOKKOS_INLINE_FUNCTION
    fp128_t imag() const { return im; }

    // Compound assignment operators
    KOKKOS_INLINE_FUNCTION
    quad_complex& operator+=(quad_complex const& other) {
        re = quad::add(re, other.re);
        im = quad::add(im, other.im);
        return *this;
    }

    KOKKOS_INLINE_FUNCTION
    quad_complex& operator-=(quad_complex const& other) {
        re = quad::sub(re, other.re);
        im = quad::sub(im, other.im);
        return *this;
    }

    KOKKOS_INLINE_FUNCTION
    quad_complex& operator*=(quad_complex const& other) {
        // (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
        fp128_t new_re = quad::sub(quad::mul(re, other.re), quad::mul(im, other.im));
        fp128_t new_im = quad::add(quad::mul(re, other.im), quad::mul(im, other.re));
        re = new_re;
        im = new_im;
        return *this;
    }

    KOKKOS_INLINE_FUNCTION
    quad_complex& operator/=(quad_complex const& other) {
        // (a + bi) / (c + di) = ((ac + bd) + (bc - ad)i) / (c^2 + d^2)
        fp128_t denom = quad::add(quad::mul(other.re, other.re), quad::mul(other.im, other.im));
        fp128_t new_re = quad::div(quad::add(quad::mul(re, other.re), quad::mul(im, other.im)), denom);
        fp128_t new_im = quad::div(quad::sub(quad::mul(im, other.re), quad::mul(re, other.im)), denom);
        re = new_re;
        im = new_im;
        return *this;
    }
    
    // Assignment from scalar (matching Kokkos::complex behavior)
    template<typename T>
    KOKKOS_INLINE_FUNCTION
    quad_complex& operator=(T val) {
        re = fp128_t(val);
        im = fp128_t(0.0q);
        return *this;
    }
};

// Arithmetic operators - only implement what is needed
KOKKOS_INLINE_FUNCTION
quad_complex operator+(quad_complex const& a, quad_complex const& b) {
    quad_complex result(quad::add(a.re, b.re), quad::add(a.im, b.im));
    return result;
}

KOKKOS_INLINE_FUNCTION
quad_complex operator-(quad_complex const& a, quad_complex const& b) {
    quad_complex result(quad::sub(a.re, b.re), quad::sub(a.im, b.im));
    return result;
}

KOKKOS_INLINE_FUNCTION
quad_complex operator*(quad_complex const& a, quad_complex const& b) {
    // (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
    quad_complex result(
        quad::sub(quad::mul(a.re, b.re), quad::mul(a.im, b.im)),
        quad::add(quad::mul(a.re, b.im), quad::mul(a.im, b.re))
    );
    return result;
}

KOKKOS_INLINE_FUNCTION
quad_complex operator/(quad_complex const& a, quad_complex const& b) {
    // (a + bi) / (c + di) = ((ac + bd) + (bc - ad)i) / (c^2 + d^2)
    fp128_t denom = quad::add(quad::mul(b.re, b.re), quad::mul(b.im, b.im));
    quad_complex result(
        quad::div(quad::add(quad::mul(a.re, b.re), quad::mul(a.im, b.im)), denom),
        quad::div(quad::sub(quad::mul(a.im, b.re), quad::mul(a.re, b.im)), denom)
    );
    return result;
}

KOKKOS_INLINE_FUNCTION
quad_complex operator-(quad_complex const& a) {
    quad_complex result(quad::neg(a.re), quad::neg(a.im));
    return result;
}

// Comparison operators
KOKKOS_INLINE_FUNCTION
bool operator==(quad_complex const& a, quad_complex const& b) {
    return (a.re == b.re) && (a.im == b.im);
}

KOKKOS_INLINE_FUNCTION
bool operator!=(quad_complex const& a, quad_complex const& b) {
    return !(a == b);
}

// Helper type trait to exclude fp128_t and quad_complex from template operators
// This prevents template operators from matching when T = fp128_t or T = quad_complex
// Uses is_fp128_type from quad_math.hpp (included above)
template<typename T>
struct is_quad_type {
    static constexpr bool value = 
        quad::is_fp128_type<T>::value || 
        std::is_same<T, quad_complex>::value;
};

// Mixed-type arithmetic operators (matching Kokkos::complex behavior)
// Promote scalars to fp128_t and return quad_complex
// Note: These templates exclude fp128_t and quad_complex to avoid redundant conversions

// Addition: scalar + quad_complex, quad_complex + scalar
template<typename T>
KOKKOS_INLINE_FUNCTION
typename std::enable_if<!is_quad_type<T>::value, quad_complex>::type
operator+(T a, quad_complex const& b) {
    fp128_t scalar = fp128_t(a);
    quad_complex result(quad::add(scalar, b.re), b.im);
    return result;
}

template<typename T>
KOKKOS_INLINE_FUNCTION
typename std::enable_if<!is_quad_type<T>::value, quad_complex>::type
operator+(quad_complex const& a, T b) {
    fp128_t scalar = fp128_t(b);
    quad_complex result(quad::add(a.re, scalar), a.im);
    return result;
}

// Subtraction: scalar - quad_complex, quad_complex - scalar
template<typename T>
KOKKOS_INLINE_FUNCTION
typename std::enable_if<!is_quad_type<T>::value, quad_complex>::type
operator-(T a, quad_complex const& b) {
    fp128_t scalar = fp128_t(a);
    quad_complex result(quad::sub(scalar, b.re), quad::neg(b.im));
    return result;
}

template<typename T>
KOKKOS_INLINE_FUNCTION
typename std::enable_if<!is_quad_type<T>::value, quad_complex>::type
operator-(quad_complex const& a, T b) {
    fp128_t scalar = fp128_t(b);
    quad_complex result(quad::sub(a.re, scalar), a.im);
    return result;
}

// Multiplication: scalar * quad_complex, quad_complex * scalar
template<typename T>
KOKKOS_INLINE_FUNCTION
typename std::enable_if<!is_quad_type<T>::value, quad_complex>::type
operator*(T a, quad_complex const& b) {
    fp128_t scalar = fp128_t(a);
    quad_complex result(quad::mul(scalar, b.re), quad::mul(scalar, b.im));
    return result;
}

template<typename T>
KOKKOS_INLINE_FUNCTION
typename std::enable_if<!is_quad_type<T>::value, quad_complex>::type
operator*(quad_complex const& a, T b) {
    fp128_t scalar = fp128_t(b);
    quad_complex result(quad::mul(a.re, scalar), quad::mul(a.im, scalar));
    return result;
}

// Division: scalar / quad_complex, quad_complex / scalar
template<typename T>
KOKKOS_INLINE_FUNCTION
typename std::enable_if<!is_quad_type<T>::value, quad_complex>::type
operator/(T a, quad_complex const& b) {
    // a / (c + di) = a * (c - di) / (c^2 + d^2)
    fp128_t scalar = fp128_t(a);
    fp128_t denom = quad::add(quad::mul(b.re, b.re), quad::mul(b.im, b.im));
    quad_complex result(
        quad::div(quad::mul(scalar, b.re), denom),
        quad::div(quad::neg(quad::mul(scalar, b.im)), denom)
    );
    return result;
}

template<typename T>
KOKKOS_INLINE_FUNCTION
typename std::enable_if<!is_quad_type<T>::value, quad_complex>::type
operator/(quad_complex const& a, T b) {
    fp128_t scalar = fp128_t(b);
    quad_complex result(quad::div(a.re, scalar), quad::div(a.im, scalar));
    return result;
}

// Explicit overloads for quad_complex op fp128_t (take precedence over templates)
// These resolve ambiguity with operator op(T, fp128_wrapper const&) from quad_math.hpp

// Multiplication: quad_complex * fp128_t
KOKKOS_INLINE_FUNCTION
quad_complex operator*(quad_complex const& a, fp128_t b) {
    quad_complex result(quad::mul(a.re, b.value), quad::mul(a.im, b.value));
    return result;
}

// Multiplication: fp128_t * quad_complex
KOKKOS_INLINE_FUNCTION
quad_complex operator*(fp128_t a, quad_complex const& b) {
    quad_complex result(quad::mul(a.value, b.re), quad::mul(a.value, b.im));
    return result;
}

// Addition: quad_complex + fp128_t
KOKKOS_INLINE_FUNCTION
quad_complex operator+(quad_complex const& a, fp128_t b) {
    quad_complex result(quad::add(a.re, b.value), a.im);
    return result;
}

// Addition: fp128_t + quad_complex
KOKKOS_INLINE_FUNCTION
quad_complex operator+(fp128_t a, quad_complex const& b) {
    quad_complex result(quad::add(a.value, b.re), b.im);
    return result;
}

// Subtraction: quad_complex - fp128_t
KOKKOS_INLINE_FUNCTION
quad_complex operator-(quad_complex const& a, fp128_t b) {
    quad_complex result(quad::sub(a.re, b.value), a.im);
    return result;
}

// Subtraction: fp128_t - quad_complex
KOKKOS_INLINE_FUNCTION
quad_complex operator-(fp128_t a, quad_complex const& b) {
    quad_complex result(quad::sub(a.value, b.re), quad::neg(b.im));
    return result;
}

// Division: quad_complex / fp128_t
KOKKOS_INLINE_FUNCTION
quad_complex operator/(quad_complex const& a, fp128_t b) {
    quad_complex result(quad::div(a.re, b.value), quad::div(a.im, b.value));
    return result;
}

// Comparison operators: scalar == quad_complex, quad_complex == scalar
// (matching Kokkos::complex behavior: compares real part, imag must be 0)
template<typename T>
KOKKOS_INLINE_FUNCTION
typename std::enable_if<!is_quad_type<T>::value, bool>::type
operator==(T a, quad_complex const& b) {
    fp128_t scalar = fp128_t(a);
    return (scalar == b.re) && (fp128_t(0.0q) == b.im);
}

template<typename T>
KOKKOS_INLINE_FUNCTION
typename std::enable_if<!is_quad_type<T>::value, bool>::type
operator==(quad_complex const& a, T b) {
    fp128_t scalar = fp128_t(b);
    return (a.re == scalar) && (a.im == fp128_t(0.0q));
}

template<typename T>
KOKKOS_INLINE_FUNCTION
typename std::enable_if<!is_quad_type<T>::value, bool>::type
operator!=(T a, quad_complex const& b) {
    return !(a == b);
}

template<typename T>
KOKKOS_INLINE_FUNCTION
typename std::enable_if<!is_quad_type<T>::value, bool>::type
operator!=(quad_complex const& a, T b) {
    return !(a == b);
}

// Absolute value (magnitude)
KOKKOS_INLINE_FUNCTION
fp128_t abs(quad_complex const& z) {
    // |z| = sqrt(re^2 + im^2)
    fp128_t re_sq = quad::mul(z.re, z.re);
    fp128_t im_sq = quad::mul(z.im, z.im);
    return quad::sqrt(quad::add(re_sq, im_sq));
}

} // namespace ql

// Function wrappers for quad_complex in ql::quad namespace
// Matching the pattern from quad_math.hpp for consistency
namespace ql {
namespace quad {

// Basic arithmetic operations for quad_complex
KOKKOS_INLINE_FUNCTION
ql::quad_complex add(ql::quad_complex const& a, ql::quad_complex const& b) {
    ql::quad_complex result(ql::quad::add(a.re, b.re), ql::quad::add(a.im, b.im));
    return result;
}

KOKKOS_INLINE_FUNCTION
ql::quad_complex sub(ql::quad_complex const& a, ql::quad_complex const& b) {
    ql::quad_complex result(ql::quad::sub(a.re, b.re), ql::quad::sub(a.im, b.im));
    return result;
}

KOKKOS_INLINE_FUNCTION
ql::quad_complex mul(ql::quad_complex const& a, ql::quad_complex const& b) {
    // (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
    ql::quad_complex result(
        ql::quad::sub(ql::quad::mul(a.re, b.re), ql::quad::mul(a.im, b.im)),
        ql::quad::add(ql::quad::mul(a.re, b.im), ql::quad::mul(a.im, b.re))
    );
    return result;
}

KOKKOS_INLINE_FUNCTION
ql::quad_complex div(ql::quad_complex const& a, ql::quad_complex const& b) {
    // (a + bi) / (c + di) = ((ac + bd) + (bc - ad)i) / (c^2 + d^2)
    fp128_t denom = ql::quad::add(ql::quad::mul(b.re, b.re), ql::quad::mul(b.im, b.im));
    ql::quad_complex result(
        ql::quad::div(ql::quad::add(ql::quad::mul(a.re, b.re), ql::quad::mul(a.im, b.im)), denom),
        ql::quad::div(ql::quad::sub(ql::quad::mul(a.im, b.re), ql::quad::mul(a.re, b.im)), denom)
    );
    return result;
}

// Math functions for quad_complex
KOKKOS_INLINE_FUNCTION
ql::quad_complex sqrt(ql::quad_complex const& z) {
    // Complex square root matching Kokkos::complex<T>::sqrt strategy:
    // Split into two branches to avoid catastrophic cancellation.
    // When re > 0: (mag + re) is well-conditioned, derive im by division.
    // When re <= 0: (mag - re) is well-conditioned, derive re by division.
    // This preserves tiny imaginary parts (e.g. from ieps prescriptions)
    // that would be destroyed by computing sqrt((mag - re)/2) directly
    // when mag ≈ re.
    fp128_t r = ql::abs(z);
    if (r == fp128_t(0.0q)) {
        ql::quad_complex result(fp128_t(0.0q), fp128_t(0.0q));
        return result;
    }
    fp128_t re_out, im_out;
    if (z.re > fp128_t(0.0q)) {
        re_out = ql::quad::sqrt(ql::quad::div(ql::quad::add(r, z.re), fp128_t(2.0q)));
        im_out = ql::quad::div(z.im, ql::quad::mul(fp128_t(2.0q), re_out));
    } else {
        im_out = ql::quad::sqrt(ql::quad::div(ql::quad::sub(r, z.re), fp128_t(2.0q)));
    if (z.im < fp128_t(0.0q)) {
            im_out = ql::quad::neg(im_out);
        }
        re_out = ql::quad::div(z.im, ql::quad::mul(fp128_t(2.0q), im_out));
    }
    ql::quad_complex result(re_out, im_out);
    return result;
}

KOKKOS_INLINE_FUNCTION
fp128_t abs(ql::quad_complex const& z) {
    // |z| = sqrt(re^2 + im^2)
    fp128_t re_sq = ql::quad::mul(z.re, z.re);
    fp128_t im_sq = ql::quad::mul(z.im, z.im);
    return ql::quad::sqrt(ql::quad::add(re_sq, im_sq));
}

KOKKOS_INLINE_FUNCTION
ql::quad_complex log(ql::quad_complex const& z) {
    // log(z) = log(|z|) + i * arg(z)
    // arg(z) = atan2(im, re)
    fp128_t mag = ql::abs(z);
    fp128_t arg = ql::quad::atan2(z.im, z.re);
    ql::quad_complex result(ql::quad::log(mag), arg);
    return result;
}

KOKKOS_INLINE_FUNCTION
ql::quad_complex neg(ql::quad_complex const& z) {
    ql::quad_complex result(ql::quad::neg(z.re), ql::quad::neg(z.im));
    return result;
}

KOKKOS_INLINE_FUNCTION
ql::quad_complex conj(ql::quad_complex const& z) {
    ql::quad_complex result(z.re, ql::quad::neg(z.im));
    return result;
}

// exp(a + bi) = exp(a) * (cos(b) + i*sin(b))
KOKKOS_INLINE_FUNCTION
ql::quad_complex exp(ql::quad_complex const& z) {
    fp128_t ea = ql::quad::exp(z.re);
    return ql::quad_complex(
        ql::quad::mul(ea, ql::quad::cos(z.im)),
        ql::quad::mul(ea, ql::quad::sin(z.im))
    );
}

// log10(z) = log(z) / log(10)
KOKKOS_INLINE_FUNCTION
ql::quad_complex log10(ql::quad_complex const& z) {
    ql::quad_complex lz = ql::quad::log(z);
    fp128_t ln10 = ql::quad::log(fp128_t(10.0));
    return ql::quad_complex(
        ql::quad::div(lz.re, ln10),
        ql::quad::div(lz.im, ln10)
    );
}

// sin(a + bi) = sin(a)*cosh(b) + i*cos(a)*sinh(b)
KOKKOS_INLINE_FUNCTION
ql::quad_complex sin(ql::quad_complex const& z) {
    return ql::quad_complex(
        ql::quad::mul(ql::quad::sin(z.re), ql::quad::cosh(z.im)),
        ql::quad::mul(ql::quad::cos(z.re), ql::quad::sinh(z.im))
    );
}

// cos(a + bi) = cos(a)*cosh(b) - i*sin(a)*sinh(b)
KOKKOS_INLINE_FUNCTION
ql::quad_complex cos(ql::quad_complex const& z) {
    return ql::quad_complex(
        ql::quad::mul(ql::quad::cos(z.re), ql::quad::cosh(z.im)),
        ql::quad::neg(ql::quad::mul(ql::quad::sin(z.re), ql::quad::sinh(z.im)))
    );
}

// tan(z) = sin(z) / cos(z)
KOKKOS_INLINE_FUNCTION
ql::quad_complex tan(ql::quad_complex const& z) {
    return ql::quad::div(ql::quad::sin(z), ql::quad::cos(z));
}

// asin(z) = -i * log(iz + sqrt(1 - z^2))
KOKKOS_INLINE_FUNCTION
ql::quad_complex asin(ql::quad_complex const& z) {
    ql::quad_complex iz(ql::quad::neg(z.im), z.re);
    ql::quad_complex z2 = ql::quad::mul(z, z);
    ql::quad_complex one_minus_z2(ql::quad::sub(fp128_t(1.0), z2.re), ql::quad::neg(z2.im));
    ql::quad_complex sq = ql::quad::sqrt(one_minus_z2);
    ql::quad_complex lg = ql::quad::log(ql::quad::add(iz, sq));
    return ql::quad_complex(lg.im, ql::quad::neg(lg.re));
}

// acos(z) = pi/2 - asin(z)
KOKKOS_INLINE_FUNCTION
ql::quad_complex acos(ql::quad_complex const& z) {
    constexpr __fp128_base pi_over_2 = __fp128_base(1.57079632679489661923132169163975144209858469968755q);
    ql::quad_complex as = ql::quad::asin(z);
    return ql::quad_complex(ql::quad::sub(fp128_t(pi_over_2), as.re), ql::quad::neg(as.im));
}

// atan(z) = i/2 * log((1 - iz) / (1 + iz))
KOKKOS_INLINE_FUNCTION
ql::quad_complex atan(ql::quad_complex const& z) {
    ql::quad_complex iz(ql::quad::neg(z.im), z.re);
    ql::quad_complex one(fp128_t(1.0), fp128_t(0.0));
    ql::quad_complex lg = ql::quad::log(ql::quad::div(ql::quad::sub(one, iz), ql::quad::add(one, iz)));
    fp128_t half(0.5);
    return ql::quad_complex(ql::quad::neg(ql::quad::mul(half, lg.im)), ql::quad::mul(half, lg.re));
}

// sinh(z) = (exp(z) - exp(-z)) / 2
KOKKOS_INLINE_FUNCTION
ql::quad_complex sinh(ql::quad_complex const& z) {
    ql::quad_complex ez  = ql::quad::exp(z);
    ql::quad_complex emz = ql::quad::exp(ql::quad::neg(z));
    ql::quad_complex d   = ql::quad::sub(ez, emz);
    fp128_t two(2.0);
    return ql::quad_complex(ql::quad::div(d.re, two), ql::quad::div(d.im, two));
}

// cosh(z) = (exp(z) + exp(-z)) / 2
KOKKOS_INLINE_FUNCTION
ql::quad_complex cosh(ql::quad_complex const& z) {
    ql::quad_complex ez  = ql::quad::exp(z);
    ql::quad_complex emz = ql::quad::exp(ql::quad::neg(z));
    ql::quad_complex s   = ql::quad::add(ez, emz);
    fp128_t two(2.0);
    return ql::quad_complex(ql::quad::div(s.re, two), ql::quad::div(s.im, two));
}

// tanh(z) = sinh(z) / cosh(z)
KOKKOS_INLINE_FUNCTION
ql::quad_complex tanh(ql::quad_complex const& z) {
    return ql::quad::div(ql::quad::sinh(z), ql::quad::cosh(z));
}

// asinh(z) = log(z + sqrt(z^2 + 1))
KOKKOS_INLINE_FUNCTION
ql::quad_complex asinh(ql::quad_complex const& z) {
    ql::quad_complex z2 = ql::quad::mul(z, z);
    ql::quad_complex z2p1(ql::quad::add(z2.re, fp128_t(1.0)), z2.im);
    return ql::quad::log(ql::quad::add(z, ql::quad::sqrt(z2p1)));
}

// acosh(z) = log(z + sqrt(z^2 - 1))
KOKKOS_INLINE_FUNCTION
ql::quad_complex acosh(ql::quad_complex const& z) {
    ql::quad_complex z2 = ql::quad::mul(z, z);
    ql::quad_complex z2m1(ql::quad::sub(z2.re, fp128_t(1.0)), z2.im);
    return ql::quad::log(ql::quad::add(z, ql::quad::sqrt(z2m1)));
}

// atanh(z) = 1/2 * log((1 + z) / (1 - z))
KOKKOS_INLINE_FUNCTION
ql::quad_complex atanh(ql::quad_complex const& z) {
    ql::quad_complex one(fp128_t(1.0), fp128_t(0.0));
    ql::quad_complex lg = ql::quad::log(ql::quad::div(ql::quad::add(one, z), ql::quad::sub(one, z)));
    fp128_t half(0.5);
    return ql::quad_complex(ql::quad::mul(half, lg.re), ql::quad::mul(half, lg.im));
}

// pow(z, w) = exp(w * log(z))
KOKKOS_INLINE_FUNCTION
ql::quad_complex pow(ql::quad_complex const& z, ql::quad_complex const& w) {
    return ql::quad::exp(ql::quad::mul(w, ql::quad::log(z)));
}

// polar(r, theta) = r*cos(theta) + i*r*sin(theta)
KOKKOS_INLINE_FUNCTION
ql::quad_complex polar(fp128_t r, fp128_t theta) {
    return ql::quad_complex(
        ql::quad::mul(r, ql::quad::cos(theta)),
        ql::quad::mul(r, ql::quad::sin(theta))
    );
}

} // namespace quad
} // namespace ql

#endif // KOKKOS_ENABLE_CUDA
