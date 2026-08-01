// Standalone smoke test for the local Kokkos __complex128 wrapper
// (impl/Kokkos_ComplexQuadPrecisionMath.hpp, applied from
// patches/kokkos_complex_quad_math.hpp — see patches/README.md).
//
// Purpose: after any rebuild of the local Kokkos install, confirm that
//   (a) the complex-quad wrapper header is present in the install tree, and
//   (b) Kokkos::<fn>((__complex128)z) is bit-exact against libquadmath ::c<fn>q.
// Kept under scripts/ so future rebuilds can reuse it. Not part of CMake.
//
// Build (adjust the include path to your Kokkos install prefix):
//   g++ -std=c++17 -fext-numeric-literals \
//       -I<kokkos-install>/include \
//       scripts/smoke_kokkos_complex_quad.cpp -lquadmath -o /tmp/smoke_cq
//   /tmp/smoke_cq
// Expected: two lines print, exit 0.

#include <impl/Kokkos_ComplexQuadPrecisionMath.hpp>
#include <quadmath.h>
#include <cstdio>

int main() {
  // z = 1 + 2i. (_Complex_I is a C-only macro not exposed in C++, so build the
  // components directly — mirrors how src/demo_complex.cpp assembles inputs.)
  __complex128 z;
  __real__ z = 1.0Q;
  __imag__ z = 2.0Q;
  __complex128 w = Kokkos::exp(z);
  __float128 re = Kokkos::real(w);
  char buf[64];
  quadmath_snprintf(buf, sizeof buf, "%.20Qe", re);
  std::printf("Re(exp(1+2i)) = %s\n", buf);

  // spot-check against ::cexpq (must be bit-exact, wrapper is a 1-line forward)
  __complex128 ref = ::cexpq(z);
  if (Kokkos::real(w) != Kokkos::real(ref) ||
      Kokkos::imag(w) != Kokkos::imag(ref)) {
    std::printf("MISMATCH: Kokkos::exp != cexpq\n");
    return 1;
  }
  std::printf("Bit-exact match with cexpq: OK\n");
  return 0;
}
