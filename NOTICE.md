# NOTICE

This repository combines Reet-authored Kokkos code (Apache-2.0) with a
C++/Kokkos port of DDFUN algorithms authored by David H. Bailey
(DHB-License, a modified-BSD-3-Clause variant).

It is therefore **dual-licensed**: most of the repository is Apache-2.0,
but the two DDFUN-derived headers carry the DHB-License, and one
Kokkos-style extension header carries Apache-2.0 WITH LLVM-exception to
match Kokkos itself. The mapping below is authoritative.

## Per-file license mapping

| File(s) | License | Notes |
|---|---|---|
| `third_party/include/dd_math.hpp` | DHB-License | C++/Kokkos port of DDFUN v04 (real). See `LICENSES/LicenseRef-DHB-License.txt`. |
| `third_party/include/dd_complex.hpp` | DHB-License | C++/Kokkos port of DDFUN v04 (complex). See `LICENSES/LicenseRef-DHB-License.txt`. |
| `patches/kokkos_complex_quad_math.hpp` | Apache-2.0 WITH LLVM-exception | Kokkos-style extension header (a companion to `Kokkos_QuadPrecisionMath.hpp`), **not** a DDFUN derivative. Licensed to match Kokkos for eventual upstream compatibility. |
| Everything else (demos, tests, harness, corpus, scripts, docs) | Apache-2.0 | Covered by the top-level `LICENSE`. |

## The DDFUN-derived files

`third_party/include/dd_math.hpp` and `third_party/include/dd_complex.hpp`
are derivative works of DDFUN v04:

- Original author: David H. Bailey (Lawrence Berkeley National Lab,
  retired / University of California, Davis).
- Original source:
  <https://www.davidhbailey.com/dhbsoftware/ddfun-v04.tar.gz>
- Original license: **DHB-License** — a modified BSD-3-Clause variant with
  a non-standard §3 grant-back clause. Copyright (c) 2024 David H. Bailey.
  All rights reserved. Full verbatim text:
  `LICENSES/LicenseRef-DHB-License.txt` (SPDX identifier
  `LicenseRef-DHB-License`) or
  <https://www.davidhbailey.com/dhbsoftware/DHB-License.txt>.

Because DDFUN is licensed under the DHB-License, this repository cannot
relicense the ported files. The port is a derivative work and inherits
DDFUN's DHB-License terms unchanged.

### DHB-License §3 grant-back — what it means for you

DHB-License §3 states that if you publish modifications or enhancements
to the DDFUN-derived files publicly (as this repository does), without a
separate written license agreement, you thereby grant David H. Bailey a
non-exclusive, royalty-free, perpetual license to install, use, modify,
prepare derivative works, incorporate into other software, distribute,
and sublicense those enhancements, in binary and source form.

In plain terms: publishing improvements to `dd_math.hpp` /
`dd_complex.hpp` publicly grants David H. Bailey a perpetual license to
incorporate those improvements back into DDFUN. Anyone who redistributes
the DDFUN-derived files from this repository inherits this term.

### Commercial-use pointer

The DDFUN website asks commercial users to contact the author. This is
not a licensing requirement in the DHB-License text itself, but it is
worth honoring: contact David H. Bailey at <dhbailey@lbl.gov> for
commercial-use questions about DDFUN.

## Contacts

- **DDFUN questions / DHB-License / commercial use:** David H. Bailey,
  <dhbailey@lbl.gov>.
- **This repository:** contact Reet (repository owner).

## Cross-references

- Full DHB-License text: `LICENSES/LicenseRef-DHB-License.txt`.
- Apache-2.0 text: `LICENSE` (mirrored at `LICENSES/Apache-2.0.txt`).
- Licensing rationale and Phase 2/3 open questions:
  `docs/TEST_SUITE_PLAN.md`, "Licensing" section.
