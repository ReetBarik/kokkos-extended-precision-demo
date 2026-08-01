# `tests/` — extended-precision test suite

This directory holds the backend-parameterized test harness for the portable
extended-precision backends benchmarked in this repo. The authoritative spec is
[`docs/TEST_SUITE_PLAN.md`](../docs/TEST_SUITE_PLAN.md) — this README is the
operational quick-reference for the harness created in **T0.1**.

## Purpose

Each backend (DD today; FF, QF in later phases) is validated across six test
layers: EFT unit tests, non-overlap invariants, property/identity tests,
differential accuracy vs a `__float128` oracle, FMA-contraction guards, and
end-to-end cancellation kernels. The harness in `test_utils.hpp` provides the
shared plumbing so a single test file can target any backend without
duplication.

As of T0.1 the only test file is `hello_test.cpp`, a smoke test that exercises
the harness end-to-end (Kokkos init, input generation, host↔device copy, oracle
comparison, reporting) on a trivial DD round-trip identity. Real coverage begins
in Phase 1.

## How to run

Configure and build (from the repo root), then run the tests:

```bash
# Configure + build (Kokkos must be installed; see top-level README/CLAUDE.md)
cmake -B build -DCMAKE_PREFIX_PATH=<kokkos-install-dir>
cmake --build build -j

# Run the test suite
ctest --test-dir build -V
```

A passing run shows:

```
    Start 1: hello_test
1/1 Test #1: hello_test .......................   Passed
```

Tests build under `build/tests/`; you can also run a binary directly, e.g.
`./build/tests/hello_test`.

## How to add a new test

Five-line recipe:

1. Create `tests/foo_test.cpp`.
2. `#include "test_utils.hpp"` and write `main()` (init Kokkos, call the runners,
   use `KOKKOS_EP_ASSERT`, `return kokkos_ep::ep_exit_code();`).
3. Add one line to `tests/CMakeLists.txt`:  `kokkos_ep_add_test(foo_test)`.
4. Rebuild: `cmake --build build -j`.
5. Run: `ctest --test-dir build -V` — `foo_test` now appears.

`kokkos_ep_add_test(<name>)` compiles `tests/<name>.cpp`, links
`Kokkos::kokkos` + the third-party include dir, applies the quadmath define, and
registers the test with CTest (with `SKIP_RETURN_CODE=77`).

## Backends and how tags map to types

The harness is templated over a **backend tag** via `BackendTraits<Tag>`:

| Tag  | Type            | Digits | Status                                  |
|------|-----------------|--------|-----------------------------------------|
| `DD` | `dd::DoubleDouble` | ~31 | supported (this branch)                 |
| `FF` | `ff::FloatFloat`   | ~14 | Phase 2 — adds `BackendTraits<FF>`      |
| `QF` | `qf::QuadFloat`    | ~29 | Phase 3 — adds `BackendTraits<QF>`      |

`DD` etc. are empty tag types, **not** the arithmetic types. `BackendTraits<DD>`
exposes `type` (`dd::DoubleDouble`), `u_squared`, `max_digits`, `name()`, and
`to_quad()`. A test written against `BackendTraits<Backend>` instantiates across
backends with no source duplication; Phase 2/3 add the FF/QF specializations.

## Framework

CTest + the lightweight `test_utils.hpp` header — **no** GoogleTest/Catch2. The
sole assertion primitive is `KOKKOS_EP_ASSERT(cond, msg)`, which prints
`file:line` + message and makes `main()` return nonzero. Rationale (and the
option to revisit in Phase 1) is documented at the top of `test_utils.hpp`.

## Graceful degradation (no LIBQUADMATH)

The `__float128` oracle comes from Kokkos's quadmath overloads, available only
when Kokkos was built with `-DKokkos_ENABLE_LIBQUADMATH=ON`. When that TPL is
absent:

- CMake still configures and the tests still **build** (no configure error).
- `KOKKOS_EP_HAVE_QUADMATH` is left undefined, so the oracle code paths compile
  out.
- Oracle-dependent tests return exit code **77** at runtime, which CTest reports
  as **`Skipped`** (not failed) via `SKIP_RETURN_CODE`.

This is the same skip-don't-fail posture T0.0/T0.3 used for the demos: a
legitimately quadmath-less Kokkos config should not turn the suite red.
