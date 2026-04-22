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

namespace quad {
namespace cuda_fp128 {

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
        re = add(re, other.re);
        im = add(im, other.im);
        return *this;
    }

    KOKKOS_INLINE_FUNCTION
    quad_complex& operator-=(quad_complex const& other) {
        re = sub(re, other.re);
        im = sub(im, other.im);
        return *this;
    }

    KOKKOS_INLINE_FUNCTION
    quad_complex& operator*=(quad_complex const& other) {
        // (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
        fp128_t new_re = sub(mul(re, other.re), mul(im, other.im));
        fp128_t new_im = add(mul(re, other.im), mul(im, other.re));
        re = new_re;
        im = new_im;
        return *this;
    }

    KOKKOS_INLINE_FUNCTION
    quad_complex& operator/=(quad_complex const& other) {
        // (a + bi) / (c + di) = ((ac + bd) + (bc - ad)i) / (c^2 + d^2)
        fp128_t denom = add(mul(other.re, other.re), mul(other.im, other.im));
        fp128_t new_re = div(add(mul(re, other.re), mul(im, other.im)), denom);
        fp128_t new_im = div(sub(mul(im, other.re), mul(re, other.im)), denom);
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
    quad_complex result(add(a.re, b.re), add(a.im, b.im));
    return result;
}

KOKKOS_INLINE_FUNCTION
quad_complex operator-(quad_complex const& a, quad_complex const& b) {
    quad_complex result(sub(a.re, b.re), sub(a.im, b.im));
    return result;
}

KOKKOS_INLINE_FUNCTION
quad_complex operator*(quad_complex const& a, quad_complex const& b) {
    // (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
    quad_complex result(
        sub(mul(a.re, b.re), mul(a.im, b.im)),
        add(mul(a.re, b.im), mul(a.im, b.re))
    );
    return result;
}

KOKKOS_INLINE_FUNCTION
quad_complex operator/(quad_complex const& a, quad_complex const& b) {
    // (a + bi) / (c + di) = ((ac + bd) + (bc - ad)i) / (c^2 + d^2)
    fp128_t denom = add(mul(b.re, b.re), mul(b.im, b.im));
    quad_complex result(
        div(add(mul(a.re, b.re), mul(a.im, b.im)), denom),
        div(sub(mul(a.im, b.re), mul(a.re, b.im)), denom)
    );
    return result;
}

KOKKOS_INLINE_FUNCTION
quad_complex operator-(quad_complex const& a) {
    quad_complex result(neg(a.re), neg(a.im));
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
        is_fp128_type<T>::value || 
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
    quad_complex result(add(scalar, b.re), b.im);
    return result;
}

template<typename T>
KOKKOS_INLINE_FUNCTION
typename std::enable_if<!is_quad_type<T>::value, quad_complex>::type
operator+(quad_complex const& a, T b) {
    fp128_t scalar = fp128_t(b);
    quad_complex result(add(a.re, scalar), a.im);
    return result;
}

// Subtraction: scalar - quad_complex, quad_complex - scalar
template<typename T>
KOKKOS_INLINE_FUNCTION
typename std::enable_if<!is_quad_type<T>::value, quad_complex>::type
operator-(T a, quad_complex const& b) {
    fp128_t scalar = fp128_t(a);
    quad_complex result(sub(scalar, b.re), neg(b.im));
    return result;
}

template<typename T>
KOKKOS_INLINE_FUNCTION
typename std::enable_if<!is_quad_type<T>::value, quad_complex>::type
operator-(quad_complex const& a, T b) {
    fp128_t scalar = fp128_t(b);
    quad_complex result(sub(a.re, scalar), a.im);
    return result;
}

// Multiplication: scalar * quad_complex, quad_complex * scalar
template<typename T>
KOKKOS_INLINE_FUNCTION
typename std::enable_if<!is_quad_type<T>::value, quad_complex>::type
operator*(T a, quad_complex const& b) {
    fp128_t scalar = fp128_t(a);
    quad_complex result(mul(scalar, b.re), mul(scalar, b.im));
    return result;
}

template<typename T>
KOKKOS_INLINE_FUNCTION
typename std::enable_if<!is_quad_type<T>::value, quad_complex>::type
operator*(quad_complex const& a, T b) {
    fp128_t scalar = fp128_t(b);
    quad_complex result(mul(a.re, scalar), mul(a.im, scalar));
    return result;
}

// Division: scalar / quad_complex, quad_complex / scalar
template<typename T>
KOKKOS_INLINE_FUNCTION
typename std::enable_if<!is_quad_type<T>::value, quad_complex>::type
operator/(T a, quad_complex const& b) {
    // a / (c + di) = a * (c - di) / (c^2 + d^2)
    fp128_t scalar = fp128_t(a);
    fp128_t denom = add(mul(b.re, b.re), mul(b.im, b.im));
    quad_complex result(
        div(mul(scalar, b.re), denom),
        div(neg(mul(scalar, b.im)), denom)
    );
    return result;
}

template<typename T>
KOKKOS_INLINE_FUNCTION
typename std::enable_if<!is_quad_type<T>::value, quad_complex>::type
operator/(quad_complex const& a, T b) {
    fp128_t scalar = fp128_t(b);
    quad_complex result(div(a.re, scalar), div(a.im, scalar));
    return result;
}

// Explicit overloads for quad_complex op fp128_t (take precedence over templates)
// These resolve ambiguity with operator op(T, fp128_wrapper const&) from quad_math.hpp

// Multiplication: quad_complex * fp128_t
KOKKOS_INLINE_FUNCTION
quad_complex operator*(quad_complex const& a, fp128_t b) {
    quad_complex result(mul(a.re, b.value), mul(a.im, b.value));
    return result;
}

// Multiplication: fp128_t * quad_complex
KOKKOS_INLINE_FUNCTION
quad_complex operator*(fp128_t a, quad_complex const& b) {
    quad_complex result(mul(a.value, b.re), mul(a.value, b.im));
    return result;
}

// Addition: quad_complex + fp128_t
KOKKOS_INLINE_FUNCTION
quad_complex operator+(quad_complex const& a, fp128_t b) {
    quad_complex result(add(a.re, b.value), a.im);
    return result;
}

// Addition: fp128_t + quad_complex
KOKKOS_INLINE_FUNCTION
quad_complex operator+(fp128_t a, quad_complex const& b) {
    quad_complex result(add(a.value, b.re), b.im);
    return result;
}

// Subtraction: quad_complex - fp128_t
KOKKOS_INLINE_FUNCTION
quad_complex operator-(quad_complex const& a, fp128_t b) {
    quad_complex result(sub(a.re, b.value), a.im);
    return result;
}

// Subtraction: fp128_t - quad_complex
KOKKOS_INLINE_FUNCTION
quad_complex operator-(fp128_t a, quad_complex const& b) {
    quad_complex result(sub(a.value, b.re), neg(b.im));
    return result;
}

// Division: quad_complex / fp128_t
KOKKOS_INLINE_FUNCTION
quad_complex operator/(quad_complex const& a, fp128_t b) {
    quad_complex result(div(a.re, b.value), div(a.im, b.value));
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
    fp128_t re_sq = mul(z.re, z.re);
    fp128_t im_sq = mul(z.im, z.im);
    return sqrt(add(re_sq, im_sq));
}

} // namespace cuda_fp128
} // namespace quad

// Function wrappers for quad_complex in quad::cuda_fp128 namespace
// Matching the pattern from quad_math.hpp for consistency
namespace quad {
namespace cuda_fp128 {

// Basic arithmetic operations for quad_complex
KOKKOS_INLINE_FUNCTION
quad_complex add(quad_complex const& a, quad_complex const& b) {
    quad_complex result(add(a.re, b.re), add(a.im, b.im));
    return result;
}

KOKKOS_INLINE_FUNCTION
quad_complex sub(quad_complex const& a, quad_complex const& b) {
    quad_complex result(sub(a.re, b.re), sub(a.im, b.im));
    return result;
}

KOKKOS_INLINE_FUNCTION
quad_complex mul(quad_complex const& a, quad_complex const& b) {
    // (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
    quad_complex result(
        sub(mul(a.re, b.re), mul(a.im, b.im)),
        add(mul(a.re, b.im), mul(a.im, b.re))
    );
    return result;
}

KOKKOS_INLINE_FUNCTION
quad_complex div(quad_complex const& a, quad_complex const& b) {
    // (a + bi) / (c + di) = ((ac + bd) + (bc - ad)i) / (c^2 + d^2)
    fp128_t denom = add(mul(b.re, b.re), mul(b.im, b.im));
    quad_complex result(
        div(add(mul(a.re, b.re), mul(a.im, b.im)), denom),
        div(sub(mul(a.im, b.re), mul(a.re, b.im)), denom)
    );
    return result;
}

// Math functions for quad_complex
KOKKOS_INLINE_FUNCTION
quad_complex sqrt(quad_complex const& z) {
    // Complex square root matching Kokkos::complex<T>::sqrt strategy:
    // Split into two branches to avoid catastrophic cancellation.
    // When re > 0: (mag + re) is well-conditioned, derive im by division.
    // When re <= 0: (mag - re) is well-conditioned, derive re by division.
    // This preserves tiny imaginary parts (e.g. from ieps prescriptions)
    // that would be destroyed by computing sqrt((mag - re)/2) directly
    // when mag ≈ re.
    fp128_t r = abs(z);
    if (r == fp128_t(0.0q)) {
        quad_complex result(fp128_t(0.0q), fp128_t(0.0q));
        return result;
    }
    fp128_t re_out, im_out;
    if (z.re > fp128_t(0.0q)) {
        re_out = sqrt(div(add(r, z.re), fp128_t(2.0q)));
        im_out = div(z.im, mul(fp128_t(2.0q), re_out));
    } else {
        im_out = sqrt(div(sub(r, z.re), fp128_t(2.0q)));
    if (z.im < fp128_t(0.0q)) {
            im_out = neg(im_out);
        }
        re_out = div(z.im, mul(fp128_t(2.0q), im_out));
    }
    quad_complex result(re_out, im_out);
    return result;
}

KOKKOS_INLINE_FUNCTION
quad_complex log(quad_complex const& z) {
    // log(z) = log(|z|) + i * arg(z)
    // arg(z) = atan2(im, re)
    fp128_t mag = abs(z);
    fp128_t arg = atan2(z.im, z.re);
    quad_complex result(log(mag), arg);
    return result;
}

KOKKOS_INLINE_FUNCTION
quad_complex neg(quad_complex const& z) {
    quad_complex result(neg(z.re), neg(z.im));
    return result;
}

KOKKOS_INLINE_FUNCTION
quad_complex conj(quad_complex const& z) {
    quad_complex result(z.re, neg(z.im));
    return result;
}

// exp(a + bi) = exp(a) * (cos(b) + i*sin(b))
KOKKOS_INLINE_FUNCTION
quad_complex exp(quad_complex const& z) {
    fp128_t ea = exp(z.re);
    return quad_complex(
        mul(ea, cos(z.im)),
        mul(ea, sin(z.im))
    );
}

// log10(z) = log(z) / log(10)
KOKKOS_INLINE_FUNCTION
quad_complex log10(quad_complex const& z) {
    quad_complex lz = log(z);
    fp128_t ln10 = log(fp128_t(10.0));
    return quad_complex(
        div(lz.re, ln10),
        div(lz.im, ln10)
    );
}

// sin(a + bi) = sin(a)*cosh(b) + i*cos(a)*sinh(b)
KOKKOS_INLINE_FUNCTION
quad_complex sin(quad_complex const& z) {
    return quad_complex(
        mul(sin(z.re), cosh(z.im)),
        mul(cos(z.re), sinh(z.im))
    );
}

// cos(a + bi) = cos(a)*cosh(b) - i*sin(a)*sinh(b)
KOKKOS_INLINE_FUNCTION
quad_complex cos(quad_complex const& z) {
    return quad_complex(
        mul(cos(z.re), cosh(z.im)),
        neg(mul(sin(z.re), sinh(z.im)))
    );
}

// tan(z) = sin(z) / cos(z)
KOKKOS_INLINE_FUNCTION
quad_complex tan(quad_complex const& z) {
    return div(sin(z), cos(z));
}

// asin(z) = -i * log(iz + sqrt(1 - z^2))
KOKKOS_INLINE_FUNCTION
quad_complex asin(quad_complex const& z) {
    quad_complex iz(neg(z.im), z.re);
    quad_complex z2 = mul(z, z);
    quad_complex one_minus_z2(sub(fp128_t(1.0), z2.re), neg(z2.im));
    quad_complex sq = sqrt(one_minus_z2);
    quad_complex lg = log(add(iz, sq));
    return quad_complex(lg.im, neg(lg.re));
}

// acos(z) = pi/2 - asin(z)
KOKKOS_INLINE_FUNCTION
quad_complex acos(quad_complex const& z) {
    constexpr __fp128_base pi_over_2 = __fp128_base(1.57079632679489661923132169163975144209858469968755q);
    quad_complex as = asin(z);
    return quad_complex(sub(fp128_t(pi_over_2), as.re), neg(as.im));
}

// atan(z) = i/2 * log((1 - iz) / (1 + iz))
KOKKOS_INLINE_FUNCTION
quad_complex atan(quad_complex const& z) {
    quad_complex iz(neg(z.im), z.re);
    quad_complex one(fp128_t(1.0), fp128_t(0.0));
    quad_complex lg = log(div(sub(one, iz), add(one, iz)));
    fp128_t half(0.5);
    return quad_complex(neg(mul(half, lg.im)), mul(half, lg.re));
}

// sinh(z) = (exp(z) - exp(-z)) / 2
KOKKOS_INLINE_FUNCTION
quad_complex sinh(quad_complex const& z) {
    quad_complex ez  = exp(z);
    quad_complex emz = exp(neg(z));
    quad_complex d   = sub(ez, emz);
    fp128_t two(2.0);
    return quad_complex(div(d.re, two), div(d.im, two));
}

// cosh(z) = (exp(z) + exp(-z)) / 2
KOKKOS_INLINE_FUNCTION
quad_complex cosh(quad_complex const& z) {
    quad_complex ez  = exp(z);
    quad_complex emz = exp(neg(z));
    quad_complex s   = add(ez, emz);
    fp128_t two(2.0);
    return quad_complex(div(s.re, two), div(s.im, two));
}

// tanh(z) = sinh(z) / cosh(z)
KOKKOS_INLINE_FUNCTION
quad_complex tanh(quad_complex const& z) {
    return div(sinh(z), cosh(z));
}

// asinh(z) = log(z + sqrt(z^2 + 1))
KOKKOS_INLINE_FUNCTION
quad_complex asinh(quad_complex const& z) {
    quad_complex z2 = mul(z, z);
    quad_complex z2p1(add(z2.re, fp128_t(1.0)), z2.im);
    return log(add(z, sqrt(z2p1)));
}

// acosh(z) = log(z + sqrt(z^2 - 1))
KOKKOS_INLINE_FUNCTION
quad_complex acosh(quad_complex const& z) {
    quad_complex z2 = mul(z, z);
    quad_complex z2m1(sub(z2.re, fp128_t(1.0)), z2.im);
    return log(add(z, sqrt(z2m1)));
}

// atanh(z) = 1/2 * log((1 + z) / (1 - z))
KOKKOS_INLINE_FUNCTION
quad_complex atanh(quad_complex const& z) {
    quad_complex one(fp128_t(1.0), fp128_t(0.0));
    quad_complex lg = log(div(add(one, z), sub(one, z)));
    fp128_t half(0.5);
    return quad_complex(mul(half, lg.re), mul(half, lg.im));
}

// pow(z, w) = exp(w * log(z))
KOKKOS_INLINE_FUNCTION
quad_complex pow(quad_complex const& z, quad_complex const& w) {
    return exp(mul(w, log(z)));
}

// polar(r, theta) = r*cos(theta) + i*r*sin(theta)
KOKKOS_INLINE_FUNCTION
quad_complex polar(fp128_t r, fp128_t theta) {
    return quad_complex(
        mul(r, cos(theta)),
        mul(r, sin(theta))
    );
}

} // namespace cuda_fp128
} // namespace quad

#endif // KOKKOS_ENABLE_CUDA
