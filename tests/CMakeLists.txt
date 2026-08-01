# ============================================================================
# tests/CMakeLists.txt — extended-precision test suite (T0.1 harness)
# ============================================================================
# Registered from the top-level CMakeLists.txt via add_subdirectory(tests),
# guarded by option KOKKOS_EP_BUILD_TESTS (default ON) and enable_testing().

# --- Oracle availability --------------------------------------------------
# The __float128 oracle (impl/Kokkos_QuadPrecisionMath.hpp) exists only when
# Kokkos was built with Kokkos_ENABLE_LIBQUADMATH=ON. Kokkos advertises enabled
# TPLs via the Kokkos_TPLS list (checked the same way in the top-level file).
# When present, we define KOKKOS_EP_HAVE_QUADMATH so test_utils.hpp compiles its
# oracle path; when absent, tests still BUILD but runtime-skip (exit code 77 ->
# CTest "Skipped"). Graceful degradation = skip, not configure error, matching
# the T0.0/T0.3 posture for the demos.
if("LIBQUADMATH" IN_LIST Kokkos_TPLS)
  set(KOKKOS_EP_QUADMATH_DEFINE KOKKOS_EP_HAVE_QUADMATH)
  message(STATUS "tests: __float128 oracle enabled (LIBQUADMATH present)")
else()
  set(KOKKOS_EP_QUADMATH_DEFINE "")
  message(WARNING
    "tests: Kokkos has no LIBQUADMATH; oracle-dependent tests will runtime-SKIP "
    "(CTest reports Skipped). Rebuild Kokkos with -DKokkos_ENABLE_LIBQUADMATH=ON "
    "to enable them.")
endif()

# --- Helper: define + register one test -----------------------------------
# Usage: kokkos_ep_add_test(<name>)  expects tests/<name>.cpp
#
# Creates target <name> from tests/<name>.cpp, links Kokkos::kokkos and the
# third-party include dir (dd_math.hpp etc.), applies the quadmath define, and
# registers it with CTest. SKIP_RETURN_CODE=77 lets a test signal "Skipped" at
# runtime (returned by main() when KOKKOS_EP_HAVE_QUADMATH is undefined).
#
# To add a new test:  create tests/foo_test.cpp, then add
#   kokkos_ep_add_test(foo_test)
# below. That's the whole recipe (see tests/README.md).
function(kokkos_ep_add_test name)
  add_executable(${name} ${name}.cpp)
  target_include_directories(${name} PRIVATE ${THIRD_PARTY_INCLUDE})
  target_link_libraries(${name} PRIVATE Kokkos::kokkos)
  if(KOKKOS_EP_QUADMATH_DEFINE)
    target_compile_definitions(${name} PRIVATE ${KOKKOS_EP_QUADMATH_DEFINE})
  endif()
  add_test(NAME ${name} COMMAND ${name})
  set_tests_properties(${name} PROPERTIES SKIP_RETURN_CODE 77)
endfunction()

# --- Helper: EFT test with FMA contraction disabled -----------------------
# Error-free-transform tests (Layer 1: T1.1 DD, T2.1 FF, T3.1 QF) validate that
# twoSum/twoProd produce an EXACT (result, error) pair. The Dekker twoProduct in
# dd_math.hpp's multiply relies on `a1*b1 - c11` being two distinct rounded
# operations; if the compiler contracts that into a single FMA the error term
# collapses to zero and the EFT silently breaks — and the test would validate a
# transform the shipped binary does not perform. So EFT test targets MUST be
# compiled with contraction OFF.
#
# This is applied PER TARGET (target_compile_options), not as a global build flag:
# only the EFT tests need it; the demos and other test layers should build with
# the project's normal flags. Per-target keeps the requirement local and explicit.
#
# Compiler coverage below matches what this project actually builds under (GCC
# host today; nvcc when Kokkos is CUDA-backed). IntelLLVM/Intel and Clang lines
# are included because they are one-liners and cost nothing, but are not the
# tested path — extend as needed for other toolchains.
function(kokkos_ep_add_eft_test name)
  kokkos_ep_add_test(${name})
  # Host C++ (GNU/Clang): -ffp-contract=off. GCC's default at -O2 is already
  # -ffp-contract=off for standard C++, but we set it explicitly so the guarantee
  # does not depend on the default or on -Ofast/-ffast-math leaking in.
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(${name} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-ffp-contract=off>)
  endif()
  # Intel compilers: -fp-model=precise disables contraction/reassociation.
  if(CMAKE_CXX_COMPILER_ID MATCHES "IntelLLVM|Intel")
    target_compile_options(${name} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-fp-model=precise>)
  endif()
  # When Kokkos is CUDA-backed, source compiles as CUDA (nvcc). --fmad=false stops
  # nvcc fusing the Dekker sequence. Guarded on COMPILE_LANGUAGE:CUDA so it is a
  # no-op for the Serial/host build; harmless when CUDA is not enabled.
  if(Kokkos_ENABLE_CUDA)
    target_compile_options(${name} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>)
  endif()
endfunction()

# --- Registered tests -----------------------------------------------------
kokkos_ep_add_test(hello_test)
kokkos_ep_add_test(corpus_test)
kokkos_ep_add_eft_test(dd_eft_test)
