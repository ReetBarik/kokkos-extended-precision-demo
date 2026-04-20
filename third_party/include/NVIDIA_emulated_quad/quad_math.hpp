//
// QCDLoop + Kokkos 2025
//
// Authors: Reet Barik      : rbarik@anl.gov
//          Taylor Childers : jchilders@anl.gov
//
// CUDA Quad Precision Math Wrapper Layer
// Isolates all CUDA quad precision functionality

#pragma once

#ifdef KOKKOS_ENABLE_CUDA

#include <crt/device_fp128_functions.h>
#include <type_traits>

namespace ql {
namespace quad {

// Forward declaration of underlying type
#ifdef __FLOAT128_CPP_SPELLING_ENABLED__
using __fp128_base = __float128;
#elif defined(__FLOAT128_C_SPELLING_ENABLED__)
using __fp128_base = _Float128;
#else
using __fp128_base = __float128;
#endif

// Wrapper struct for __float128 with operator overloading
// Enables native operator syntax (a + b) while using __nv_fp128_* functions
struct fp128_wrapper {
    __fp128_base value;
    
    KOKKOS_INLINE_FUNCTION
    fp128_wrapper() = default;
    
    KOKKOS_INLINE_FUNCTION constexpr
    fp128_wrapper(__fp128_base v) : value(v) {}
    
    // Explicit constructors for common scalar types
    // Use multiplication with 1.0q to force conversion via arithmetic
    KOKKOS_INLINE_FUNCTION
    fp128_wrapper(int v) : value(__fp128_base(v * 1.0q)) {}
    
    KOKKOS_INLINE_FUNCTION
    fp128_wrapper(double v) : value(__fp128_base(v * 1.0q)) {}
    
    // Explicit copy constructor - provides NVCC with an actual function body
    // to emit for device code (= default fails to synthesize device-side symbols)
    KOKKOS_INLINE_FUNCTION
    fp128_wrapper(const fp128_wrapper& other) : value(other.value) {}
    
    // Explicit copy assignment operator - same rationale as copy constructor
    KOKKOS_INLINE_FUNCTION
    fp128_wrapper& operator=(const fp128_wrapper& other) { value = other.value; return *this; }
    
    // Destructor - use default to maintain trivial copyability
    KOKKOS_INLINE_FUNCTION
    ~fp128_wrapper() = default;
    
    // Implicit conversion to underlying type
    KOKKOS_INLINE_FUNCTION constexpr
    operator __fp128_base() const { return value; }
    
    // Arithmetic operators
    KOKKOS_INLINE_FUNCTION
    fp128_wrapper operator+(fp128_wrapper const& other) const {
#ifdef __CUDA_ARCH__
        fp128_wrapper result(__nv_fp128_add(value, other.value));
        return result;
#else
        fp128_wrapper result(value + other.value);  // Host: native operator
        return result;
#endif
    }
    
    KOKKOS_INLINE_FUNCTION
    fp128_wrapper operator-(fp128_wrapper const& other) const {
#ifdef __CUDA_ARCH__
        fp128_wrapper result(__nv_fp128_sub(value, other.value));
        return result;
#else
        fp128_wrapper result(value - other.value);
        return result;
#endif
    }
    
    KOKKOS_INLINE_FUNCTION
    fp128_wrapper operator*(fp128_wrapper const& other) const {
#ifdef __CUDA_ARCH__
        fp128_wrapper result(__nv_fp128_mul(value, other.value));
        return result;
#else
        fp128_wrapper result(value * other.value);
        return result;
#endif
    }
    
    KOKKOS_INLINE_FUNCTION
    fp128_wrapper operator/(fp128_wrapper const& other) const {
#ifdef __CUDA_ARCH__
        fp128_wrapper result(__nv_fp128_div(value, other.value));
        return result;
#else
        fp128_wrapper result(value / other.value);
        return result;
#endif
    }
    
    // Unary operators
    KOKKOS_INLINE_FUNCTION
    fp128_wrapper operator-() const {
#ifdef __CUDA_ARCH__
        // Use subtraction from zero
        fp128_wrapper result(__nv_fp128_sub(__fp128_base(0.0q), value));
        return result;
#else
        fp128_wrapper result(-value);
        return result;
#endif
    }
    
    // Compound assignment operators - optimized to avoid temporary creation
    // Directly modify value member to prevent copy construction of temporaries
    KOKKOS_INLINE_FUNCTION
    fp128_wrapper& operator+=(fp128_wrapper const& other) {
#ifdef __CUDA_ARCH__
        value = __nv_fp128_add(value, other.value);
#else
        value = value + other.value;  // Host: native operator
#endif
        return *this;
    }
    
    KOKKOS_INLINE_FUNCTION
    fp128_wrapper& operator-=(fp128_wrapper const& other) {
#ifdef __CUDA_ARCH__
        value = __nv_fp128_sub(value, other.value);
#else
        value = value - other.value;  // Host: native operator
#endif
        return *this;
    }
    
    KOKKOS_INLINE_FUNCTION
    fp128_wrapper& operator*=(fp128_wrapper const& other) {
#ifdef __CUDA_ARCH__
        value = __nv_fp128_mul(value, other.value);
#else
        value = value * other.value;  // Host: native operator
#endif
        return *this;
    }
    
    KOKKOS_INLINE_FUNCTION
    fp128_wrapper& operator/=(fp128_wrapper const& other) {
#ifdef __CUDA_ARCH__
        value = __nv_fp128_div(value, other.value);
#else
        value = value / other.value;  // Host: native operator
#endif
        return *this;
    }
    
    // Comparison operators
    KOKKOS_INLINE_FUNCTION
    bool operator==(fp128_wrapper const& other) const {
        return value == other.value;
    }
    
    KOKKOS_INLINE_FUNCTION
    bool operator!=(fp128_wrapper const& other) const {
        return value != other.value;
    }
    
    KOKKOS_INLINE_FUNCTION
    bool operator<(fp128_wrapper const& other) const {
        return value < other.value;
    }
    
    KOKKOS_INLINE_FUNCTION
    bool operator<=(fp128_wrapper const& other) const {
        return value <= other.value;
    }
    
    KOKKOS_INLINE_FUNCTION
    bool operator>(fp128_wrapper const& other) const {
        return value > other.value;
    }
    
    KOKKOS_INLINE_FUNCTION
    bool operator>=(fp128_wrapper const& other) const {
        return value >= other.value;
    }
};

// Type alias: fp128_t is now a wrapper struct with operator overloading
using fp128_t = fp128_wrapper;

} // namespace quad

// Forward declaration of quad_complex (defined in quad_complex.hpp)
// Needed to exclude it from template operators without circular dependency
struct quad_complex;

namespace quad {

// Helper type trait to exclude fp128_t and quad_complex from template operators
// This prevents template operators from matching when T = fp128_t, fp128_wrapper, or quad_complex
template<typename T>
struct is_fp128_type {
    static constexpr bool value = 
        std::is_same<T, fp128_t>::value || 
        std::is_same<T, fp128_wrapper>::value ||
        std::is_same<T, ql::quad_complex>::value;
};

// Mixed-type arithmetic operators (matching Kokkos::complex behavior)
// Always promote scalars to fp128_t (highest precision)
// Note: These templates exclude fp128_t to avoid redundant conversions

// Addition: fp128_t + scalar
template<typename T>
KOKKOS_INLINE_FUNCTION
typename std::enable_if<!is_fp128_type<T>::value, fp128_wrapper>::type
operator+(fp128_wrapper const& a, T b) {
    fp128_wrapper result = a + fp128_wrapper(fp128_t(b));
    return result;
}

template<typename T>
KOKKOS_INLINE_FUNCTION
typename std::enable_if<!is_fp128_type<T>::value, fp128_wrapper>::type
operator+(T a, fp128_wrapper const& b) {
    fp128_wrapper result = fp128_wrapper(fp128_t(a)) + b;
    return result;
}

// Subtraction: fp128_t - scalar
template<typename T>
KOKKOS_INLINE_FUNCTION
typename std::enable_if<!is_fp128_type<T>::value, fp128_wrapper>::type
operator-(fp128_wrapper const& a, T b) {
    fp128_wrapper result = a - fp128_wrapper(fp128_t(b));
    return result;
}

template<typename T>
KOKKOS_INLINE_FUNCTION
typename std::enable_if<!is_fp128_type<T>::value, fp128_wrapper>::type
operator-(T a, fp128_wrapper const& b) {
    fp128_wrapper result = fp128_wrapper(fp128_t(a)) - b;
    return result;
}

// Multiplication: fp128_t * scalar
template<typename T>
KOKKOS_INLINE_FUNCTION
typename std::enable_if<!is_fp128_type<T>::value, fp128_wrapper>::type
operator*(fp128_wrapper const& a, T b) {
    fp128_wrapper result = a * fp128_wrapper(fp128_t(b));
    return result;
}

template<typename T>
KOKKOS_INLINE_FUNCTION
typename std::enable_if<!is_fp128_type<T>::value, fp128_wrapper>::type
operator*(T a, fp128_wrapper const& b) {
    fp128_wrapper result = fp128_wrapper(fp128_t(a)) * b;
    return result;
}

// Division: fp128_t / scalar
template<typename T>
KOKKOS_INLINE_FUNCTION
typename std::enable_if<!is_fp128_type<T>::value, fp128_wrapper>::type
operator/(fp128_wrapper const& a, T b) {
    fp128_wrapper result = a / fp128_wrapper(fp128_t(b));
    return result;
}

template<typename T>
KOKKOS_INLINE_FUNCTION
typename std::enable_if<!is_fp128_type<T>::value, fp128_wrapper>::type
operator/(T a, fp128_wrapper const& b) {
    fp128_wrapper result = fp128_wrapper(fp128_t(a)) / b;
    return result;
}

// Comparison operators: fp128_t op scalar
template<typename T>
KOKKOS_INLINE_FUNCTION
typename std::enable_if<!is_fp128_type<T>::value, bool>::type
operator==(fp128_wrapper const& a, T b) {
    return a == fp128_wrapper(fp128_t(b));
}

template<typename T>
KOKKOS_INLINE_FUNCTION
typename std::enable_if<!is_fp128_type<T>::value, bool>::type
operator==(T a, fp128_wrapper const& b) {
    return fp128_wrapper(fp128_t(a)) == b;
}

template<typename T>
KOKKOS_INLINE_FUNCTION
typename std::enable_if<!is_fp128_type<T>::value, bool>::type
operator!=(fp128_wrapper const& a, T b) {
    return a != fp128_wrapper(fp128_t(b));
}

template<typename T>
KOKKOS_INLINE_FUNCTION
typename std::enable_if<!is_fp128_type<T>::value, bool>::type
operator!=(T a, fp128_wrapper const& b) {
    return fp128_wrapper(fp128_t(a)) != b;
}

template<typename T>
KOKKOS_INLINE_FUNCTION
typename std::enable_if<!is_fp128_type<T>::value, bool>::type
operator<(fp128_wrapper const& a, T b) {
    return a < fp128_wrapper(fp128_t(b));
}

template<typename T>
KOKKOS_INLINE_FUNCTION
typename std::enable_if<!is_fp128_type<T>::value, bool>::type
operator<(T a, fp128_wrapper const& b) {
    return fp128_wrapper(fp128_t(a)) < b;
}

template<typename T>
KOKKOS_INLINE_FUNCTION
typename std::enable_if<!is_fp128_type<T>::value, bool>::type
operator<=(fp128_wrapper const& a, T b) {
    return a <= fp128_wrapper(fp128_t(b));
}

template<typename T>
KOKKOS_INLINE_FUNCTION
typename std::enable_if<!is_fp128_type<T>::value, bool>::type
operator<=(T a, fp128_wrapper const& b) {
    return fp128_wrapper(fp128_t(a)) <= b;
}

template<typename T>
KOKKOS_INLINE_FUNCTION
typename std::enable_if<!is_fp128_type<T>::value, bool>::type
operator>(fp128_wrapper const& a, T b) {
    return a > fp128_wrapper(fp128_t(b));
}

template<typename T>
KOKKOS_INLINE_FUNCTION
typename std::enable_if<!is_fp128_type<T>::value, bool>::type
operator>(T a, fp128_wrapper const& b) {
    return fp128_wrapper(fp128_t(a)) > b;
}

template<typename T>
KOKKOS_INLINE_FUNCTION
typename std::enable_if<!is_fp128_type<T>::value, bool>::type
operator>=(fp128_wrapper const& a, T b) {
    return a >= fp128_wrapper(fp128_t(b));
}

template<typename T>
KOKKOS_INLINE_FUNCTION
typename std::enable_if<!is_fp128_type<T>::value, bool>::type
operator>=(T a, fp128_wrapper const& b) {
    return fp128_wrapper(fp128_t(a)) >= b;
}

// Basic arithmetic operations - delegate to operators
KOKKOS_INLINE_FUNCTION
fp128_t add(fp128_t a, fp128_t b) {
    return a + b;  // Uses fp128_wrapper::operator+
}

KOKKOS_INLINE_FUNCTION
fp128_t sub(fp128_t a, fp128_t b) {
    return a - b;  // Uses fp128_wrapper::operator-
}

KOKKOS_INLINE_FUNCTION
fp128_t mul(fp128_t a, fp128_t b) {
    return a * b;  // Uses fp128_wrapper::operator*
}

KOKKOS_INLINE_FUNCTION
fp128_t div(fp128_t a, fp128_t b) {
    return a / b;  // Uses fp128_wrapper::operator/
}

// Math functions
KOKKOS_INLINE_FUNCTION
fp128_t sqrt(fp128_t x) {
#ifdef __CUDA_ARCH__
    return fp128_t(__nv_fp128_sqrt(x.value));
#else
    return fp128_t(0.0q);  // Device-only function
#endif
}

KOKKOS_INLINE_FUNCTION
fp128_t abs(fp128_t x) {
#ifdef __CUDA_ARCH__
    return fp128_t(__nv_fp128_fabs(x.value));
#else
    return fp128_t(0.0q);  // Device-only function
#endif
}

KOKKOS_INLINE_FUNCTION
fp128_t log(fp128_t x) {
#ifdef __CUDA_ARCH__
    return fp128_t(__nv_fp128_log(x.value));
#else
    return fp128_t(0.0q);  // Device-only function
#endif
}

KOKKOS_INLINE_FUNCTION
fp128_t exp(fp128_t x) {
#ifdef __CUDA_ARCH__
    return fp128_t(__nv_fp128_exp(x.value));
#else
    return fp128_t(0.0q);  // Device-only function
#endif
}

KOKKOS_INLINE_FUNCTION
fp128_t pow(fp128_t x, fp128_t y) {
#ifdef __CUDA_ARCH__
    return fp128_t(__nv_fp128_pow(x.value, y.value));
#else
    return fp128_t(0.0q);  // Device-only function
#endif
}

KOKKOS_INLINE_FUNCTION
fp128_t acosh(fp128_t x) {
#ifdef __CUDA_ARCH__
    return fp128_t(__nv_fp128_acosh(x.value));
#else
    return fp128_t(0.0q);  // Device-only function
#endif
}

KOKKOS_INLINE_FUNCTION
fp128_t asinh(fp128_t x) {
#ifdef __CUDA_ARCH__
    return fp128_t(__nv_fp128_asinh(x.value));
#else
    return fp128_t(0.0q);  // Device-only function
#endif
}

KOKKOS_INLINE_FUNCTION
fp128_t atanh(fp128_t x) {
#ifdef __CUDA_ARCH__
    return fp128_t(__nv_fp128_atanh(x.value));
#else
    return fp128_t(0.0q);  // Device-only function
#endif
}

KOKKOS_INLINE_FUNCTION
fp128_t neg(fp128_t x) {
    return -x;  // Uses fp128_wrapper::operator- (unary)
}

KOKKOS_INLINE_FUNCTION
fp128_t sin(fp128_t x) {
#ifdef __CUDA_ARCH__
    return fp128_t(__nv_fp128_sin(x.value));
#else
    return fp128_t(0.0q);
#endif
}

KOKKOS_INLINE_FUNCTION
fp128_t cos(fp128_t x) {
#ifdef __CUDA_ARCH__
    return fp128_t(__nv_fp128_cos(x.value));
#else
    return fp128_t(0.0q);
#endif
}

KOKKOS_INLINE_FUNCTION
fp128_t tan(fp128_t x) {
#ifdef __CUDA_ARCH__
    return fp128_t(__nv_fp128_tan(x.value));
#else
    return fp128_t(0.0q);
#endif
}

KOKKOS_INLINE_FUNCTION
fp128_t asin(fp128_t x) {
#ifdef __CUDA_ARCH__
    return fp128_t(__nv_fp128_asin(x.value));
#else
    return fp128_t(0.0q);
#endif
}

KOKKOS_INLINE_FUNCTION
fp128_t acos(fp128_t x) {
#ifdef __CUDA_ARCH__
    return fp128_t(__nv_fp128_acos(x.value));
#else
    return fp128_t(0.0q);
#endif
}

KOKKOS_INLINE_FUNCTION
fp128_t atan(fp128_t x) {
#ifdef __CUDA_ARCH__
    return fp128_t(__nv_fp128_atan(x.value));
#else
    return fp128_t(0.0q);
#endif
}

KOKKOS_INLINE_FUNCTION
fp128_t sinh(fp128_t x) {
#ifdef __CUDA_ARCH__
    return fp128_t(__nv_fp128_sinh(x.value));
#else
    return fp128_t(0.0q);
#endif
}

KOKKOS_INLINE_FUNCTION
fp128_t cosh(fp128_t x) {
#ifdef __CUDA_ARCH__
    return fp128_t(__nv_fp128_cosh(x.value));
#else
    return fp128_t(0.0q);
#endif
}

KOKKOS_INLINE_FUNCTION
fp128_t tanh(fp128_t x) {
#ifdef __CUDA_ARCH__
    return fp128_t(__nv_fp128_tanh(x.value));
#else
    return fp128_t(0.0q);
#endif
}

KOKKOS_INLINE_FUNCTION
fp128_t exp2(fp128_t x) {
#ifdef __CUDA_ARCH__
    return fp128_t(__nv_fp128_exp2(x.value));
#else
    return fp128_t(0.0q);
#endif
}

KOKKOS_INLINE_FUNCTION
fp128_t exp10(fp128_t x) {
#ifdef __CUDA_ARCH__
    return fp128_t(__nv_fp128_exp10(x.value));
#else
    return fp128_t(0.0q);
#endif
}

KOKKOS_INLINE_FUNCTION
fp128_t expm1(fp128_t x) {
#ifdef __CUDA_ARCH__
    return fp128_t(__nv_fp128_expm1(x.value));
#else
    return fp128_t(0.0q);
#endif
}

KOKKOS_INLINE_FUNCTION
fp128_t log2(fp128_t x) {
#ifdef __CUDA_ARCH__
    return fp128_t(__nv_fp128_log2(x.value));
#else
    return fp128_t(0.0q);
#endif
}

KOKKOS_INLINE_FUNCTION
fp128_t log10(fp128_t x) {
#ifdef __CUDA_ARCH__
    return fp128_t(__nv_fp128_log10(x.value));
#else
    return fp128_t(0.0q);
#endif
}

KOKKOS_INLINE_FUNCTION
fp128_t log1p(fp128_t x) {
#ifdef __CUDA_ARCH__
    return fp128_t(__nv_fp128_log1p(x.value));
#else
    return fp128_t(0.0q);
#endif
}

KOKKOS_INLINE_FUNCTION
fp128_t hypot(fp128_t x, fp128_t y) {
#ifdef __CUDA_ARCH__
    return fp128_t(__nv_fp128_hypot(x.value, y.value));
#else
    return fp128_t(0.0q);
#endif
}

KOKKOS_INLINE_FUNCTION
fp128_t fma(fp128_t x, fp128_t y, fp128_t z) {
#ifdef __CUDA_ARCH__
    return fp128_t(__nv_fp128_fma(x.value, y.value, z.value));
#else
    return fp128_t(0.0q);
#endif
}

KOKKOS_INLINE_FUNCTION
fp128_t fmod(fp128_t x, fp128_t y) {
#ifdef __CUDA_ARCH__
    return fp128_t(__nv_fp128_fmod(x.value, y.value));
#else
    return fp128_t(0.0q);
#endif
}

KOKKOS_INLINE_FUNCTION
fp128_t remainder(fp128_t x, fp128_t y) {
#ifdef __CUDA_ARCH__
    return fp128_t(__nv_fp128_remainder(x.value, y.value));
#else
    return fp128_t(0.0q);
#endif
}

KOKKOS_INLINE_FUNCTION
fp128_t copysign(fp128_t x, fp128_t y) {
#ifdef __CUDA_ARCH__
    return fp128_t(__nv_fp128_copysign(x.value, y.value));
#else
    return fp128_t(0.0q);
#endif
}

KOKKOS_INLINE_FUNCTION
fp128_t fmax(fp128_t x, fp128_t y) {
#ifdef __CUDA_ARCH__
    return fp128_t(__nv_fp128_fmax(x.value, y.value));
#else
    return fp128_t(0.0q);
#endif
}

KOKKOS_INLINE_FUNCTION
fp128_t fmin(fp128_t x, fp128_t y) {
#ifdef __CUDA_ARCH__
    return fp128_t(__nv_fp128_fmin(x.value, y.value));
#else
    return fp128_t(0.0q);
#endif
}

KOKKOS_INLINE_FUNCTION
fp128_t fdim(fp128_t x, fp128_t y) {
#ifdef __CUDA_ARCH__
    return fp128_t(__nv_fp128_fdim(x.value, y.value));
#else
    return fp128_t(0.0q);
#endif
}

KOKKOS_INLINE_FUNCTION
fp128_t ceil(fp128_t x) {
#ifdef __CUDA_ARCH__
    return fp128_t(__nv_fp128_ceil(x.value));
#else
    return fp128_t(0.0q);
#endif
}

KOKKOS_INLINE_FUNCTION
fp128_t floor(fp128_t x) {
#ifdef __CUDA_ARCH__
    return fp128_t(__nv_fp128_floor(x.value));
#else
    return fp128_t(0.0q);
#endif
}

KOKKOS_INLINE_FUNCTION
fp128_t round(fp128_t x) {
#ifdef __CUDA_ARCH__
    return fp128_t(__nv_fp128_round(x.value));
#else
    return fp128_t(0.0q);
#endif
}

KOKKOS_INLINE_FUNCTION
fp128_t trunc(fp128_t x) {
#ifdef __CUDA_ARCH__
    return fp128_t(__nv_fp128_trunc(x.value));
#else
    return fp128_t(0.0q);
#endif
}

KOKKOS_INLINE_FUNCTION
fp128_t rint(fp128_t x) {
#ifdef __CUDA_ARCH__
    return fp128_t(__nv_fp128_rint(x.value));
#else
    return fp128_t(0.0q);
#endif
}

KOKKOS_INLINE_FUNCTION
fp128_t ldexp(fp128_t x, int exp) {
#ifdef __CUDA_ARCH__
    return fp128_t(__nv_fp128_ldexp(x.value, exp));
#else
    return fp128_t(0.0q);
#endif
}

KOKKOS_INLINE_FUNCTION
fp128_t atan2(fp128_t y, fp128_t x) {
#ifdef __CUDA_ARCH__
    // atan2 is not directly available in CUDA fp128, implement using atan
    // atan2(y, x) = atan(y/x) with quadrant adjustment
    if (x.value != __fp128_base(0.0q)) {
        __fp128_base ratio = __nv_fp128_div(y.value, x.value);
        __fp128_base atan_val = __nv_fp128_atan(ratio);
        if (x.value < __fp128_base(0.0q)) {
            constexpr __fp128_base pi = __fp128_base(3.14159265358979323846264338327950288419716939937510q);
            if (y.value >= __fp128_base(0.0q)) {
                return fp128_t(__nv_fp128_add(atan_val, pi));
            } else {
                return fp128_t(__nv_fp128_sub(atan_val, pi));
            }
        }
        return fp128_t(atan_val);
    } else {
        constexpr __fp128_base pi_over_2 = __fp128_base(1.57079632679489661923132169163975144209858469968755q);
        if (y.value > __fp128_base(0.0q)) {
            return fp128_t(pi_over_2);  // pi/2
        } else if (y.value < __fp128_base(0.0q)) {
            return fp128_t(-pi_over_2);  // -pi/2
        } else {
            return fp128_t(0.0q);  // 0/0 case
        }
    }
#else
    return fp128_t(0.0q);  // Device-only function
#endif
}

} // namespace quad

// Type alias for convenience (matches examples)
using fp128_t = quad::fp128_t;

} // namespace ql

// Note: fp128_t is now a wrapper struct (fp128_wrapper) that enables:
// 1. Operator overloading (a + b, a - b, etc.) using __nv_fp128_* functions
// 2. Backward compatibility via wrapper functions (add, sub, mul, div)
// 3. Transparent usage - fp128_t can be used wherever __float128 is expected
// 4. Both operator syntax (a + b) and function syntax (quad::add(a, b)) work

#endif // KOKKOS_ENABLE_CUDA
