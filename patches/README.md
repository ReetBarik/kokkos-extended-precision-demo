# Local Kokkos patches

## `kokkos_complex_quad_math.hpp`

### What it is

A local extension to Kokkos's `core/src/impl/Kokkos_QuadPrecisionMath.hpp`.
That upstream header provides `__float128` overloads in `namespace Kokkos`
(`Kokkos::exp`, `Kokkos::sin`, …) so a host-side quadmath oracle can be spelled
in Kokkos terms. This patch adds the **`__complex128`** counterparts
(`Kokkos::exp`, `Kokkos::sqrt`, `Kokkos::conj`, `Kokkos::real`, …), which Kokkos
does not ship.

Each overload is a one-line forward to the corresponding `libquadmath`
`::c<fn>q` function (e.g. `Kokkos::exp((__complex128)z)` → `::cexpq(z)`), so it is
**bit-exact against `::c<fn>q` by construction**.

Overloads provided (namespace `Kokkos`):

- `abs`, `real`, `imag` (return `__float128`)
- `conj`
- `exp`, `log`, `log10`
- `pow`, `sqrt`
- `sin`, `cos`, `tan`, `asin`, `acos`, `atan`
- `sinh`, `cosh`, `tanh`, `asinh`, `acosh`, `atanh`

This set is exactly what `src/demo_complex.cpp` needs for its `__complex128`
oracle. `arg` (`::cargq`) is intentionally omitted because the demo does not use
it; add it if a future consumer needs it.

### It is NOT upstream

This header is not part of Kokkos. It is carried locally in this repo for
reproducibility and applied to the local Kokkos install used to build the demos.

### Why not upstream (yet)

Reet's call: keep it local until the extended-precision test suite stabilizes.
Once the oracle surface (real + complex) is settled by the test suite, this can
be proposed upstream as a companion to `Kokkos_QuadPrecisionMath.hpp`.

### How to apply

Copy the header into your Kokkos **source** tree, then rebuild/reinstall Kokkos
with libquadmath enabled:

```bash
cp patches/kokkos_complex_quad_math.hpp \
   <kokkos-source>/core/src/impl/Kokkos_ComplexQuadPrecisionMath.hpp

# reconfigure/rebuild Kokkos with libquadmath:
cmake -S <kokkos-source> -B <kokkos-build> \
      -DCMAKE_INSTALL_PREFIX=<kokkos-install> \
      -DKokkos_ENABLE_LIBQUADMATH=ON <other-flags>
cmake --build <kokkos-build> -j --target install
```

Then confirm it landed in the install tree:

```bash
ls <kokkos-install>/include/impl/Kokkos_ComplexQuadPrecisionMath.hpp
```

If your Kokkos build does not install `impl/` headers automatically, copy the
header directly into the install tree:

```bash
cp patches/kokkos_complex_quad_math.hpp \
   <kokkos-install>/include/impl/Kokkos_ComplexQuadPrecisionMath.hpp
```

A standalone verifier lives at `scripts/smoke_kokkos_complex_quad.cpp`; compile
and run it against the install tree to confirm the wrapper is present and
bit-exact against `::cexpq`.

### Tested against

- Kokkos 5.1.0 (`KOKKOS_VERSION 50100`), built with
  `-DKokkos_ENABLE_LIBQUADMATH=ON`, Serial backend, GCC 13.3.0.
- Install prefix used for the demos: `/home/rbarik/kokkos-install-quadmath`.

### License

`kokkos_complex_quad_math.hpp` is licensed **Apache-2.0 WITH
LLVM-exception**, matching Kokkos's own license — carried in its SPDX
header:

```
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project
```

This is deliberate: the file is a Kokkos-style extension header (a
companion to `Kokkos_QuadPrecisionMath.hpp`), **not** a DDFUN derivative.
Licensing it to match Kokkos means it can be upstreamed verbatim if a
Kokkos PR is ever opened, with no relicensing step.

This is separate from the DDFUN-derived files
(`third_party/include/dd_math.hpp`, `dd_complex.hpp`), which are C++/Kokkos
ports of DDFUN and carry the **DHB-License** instead. See the top-level
`NOTICE.md` for the full per-file license mapping and
`docs/TEST_SUITE_PLAN.md` "Licensing" section for the rationale.
