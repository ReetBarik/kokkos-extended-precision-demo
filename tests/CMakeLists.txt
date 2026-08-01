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
  # Tell the test source it is the contraction-OFF variant (gate on mismatches).
  # dd_fma_guard_test.cpp keys its gate-vs-report behavior on this; other EFT
  # tests ignore it. Harmless where unused.
  target_compile_definitions(${name} PRIVATE KOKKOS_EP_CONTRACTION_MODE=0)
endfunction()

# --- Helper: companion EFT test with FMA contraction forced ON ------------
# T1.5's positive counterpart to kokkos_ep_add_eft_test. It builds the SAME test
# source into a SEPARATE executable (<name>_contract_on) compiled with contraction
# forced ON, so the Dekker twoProduct is exercised under the posture the OFF helper
# protects against. Whether that produces the same (compiler didn't contract) or
# different (compiler did contract) results, the outcome is recorded rather than
# gated — the ON test is a REPORTER (see dd_fma_guard_test.cpp header). Both
# variants coexist because the target name gets the _contract_on suffix.
#
# We do not call kokkos_ep_add_test here (it would collide on the base target name
# and apply no contraction flags); instead we build the suffixed target directly,
# mirroring kokkos_ep_add_test's body plus the ON flags/definitions.
function(kokkos_ep_add_eft_test_contract_on name)
  set(tgt ${name}_contract_on)
  add_executable(${tgt} ${name}.cpp)
  target_include_directories(${tgt} PRIVATE ${THIRD_PARTY_INCLUDE})
  target_link_libraries(${tgt} PRIVATE Kokkos::kokkos)
  if(KOKKOS_EP_QUADMATH_DEFINE)
    target_compile_definitions(${tgt} PRIVATE ${KOKKOS_EP_QUADMATH_DEFINE})
  endif()
  add_test(NAME ${tgt} COMMAND ${tgt})
  set_tests_properties(${tgt} PROPERTIES SKIP_RETURN_CODE 77)

  # Force contraction ON (the mirror image of kokkos_ep_add_eft_test).
  # GCC/Clang: -ffp-contract=fast is the portable spelling both accept (GCC also
  # accepts =on, but =fast is the one clang honors identically, so we use =fast for
  # both). This permits — does not force — the compiler to fuse mul-then-add; on a
  # non-FMA ISA target GCC still emits mul+sub, which is exactly the "belt+
  # suspenders" outcome the ON reporter records.
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(${tgt} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-ffp-contract=fast>)
  endif()
  # Intel: -fp-model=fast enables contraction/reassociation.
  if(CMAKE_CXX_COMPILER_ID MATCHES "IntelLLVM|Intel")
    target_compile_options(${tgt} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-fp-model=fast>)
  endif()
  # CUDA: --fmad=true is nvcc's DEFAULT; stated explicitly for clarity/symmetry.
  if(Kokkos_ENABLE_CUDA)
    target_compile_options(${tgt} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=true>)
  endif()

  # Contraction-ON mode (report, don't gate) + baseline path for drift detection.
  target_compile_definitions(${tgt} PRIVATE
    KOKKOS_EP_CONTRACTION_MODE=1
    KOKKOS_EP_BASELINE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/dd_fma_guard_baseline.txt")
endfunction()

# --- Registered tests -----------------------------------------------------
kokkos_ep_add_test(hello_test)
kokkos_ep_add_test(corpus_test)
# T1.2 non-overlap invariant: oracle-independent, no contraction flags needed.
kokkos_ep_add_test(dd_invariant_test)
# T1.3 property/identity tests: Group A bit-exact (no oracle), Group B/Test C use
# the __float128 oracle. Plain helper — NOT an EFT test, so no contraction flags.
kokkos_ep_add_test(dd_property_test)
kokkos_ep_add_eft_test(dd_eft_test)
# T1.5 FMA-contraction guard: same source, two postures (see dd_fma_guard_test.cpp).
kokkos_ep_add_eft_test(dd_fma_guard_test)              # -> dd_fma_guard_test              (OFF, gates)
kokkos_ep_add_eft_test_contract_on(dd_fma_guard_test)  # -> dd_fma_guard_test_contract_on  (ON, reports)
