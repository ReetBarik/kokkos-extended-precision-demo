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

# --- Registered tests -----------------------------------------------------
kokkos_ep_add_test(hello_test)
kokkos_ep_add_test(corpus_test)
