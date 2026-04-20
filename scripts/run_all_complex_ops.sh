#!/usr/bin/env bash
# Run kokkos_ep_demo_complex over all complex operations in one table.
# Extra arguments are forwarded (e.g. --batch 100000 --repeats 5 --seed 42).
#
# Usage:
#   ./scripts/run_all_complex_ops.sh [extra args...]
#   KOKKOS_EP_DEMO_COMPLEX=/path/to/kokkos_ep_demo_complex ./scripts/run_all_complex_ops.sh --repeats 3

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
EXE="${KOKKOS_EP_DEMO_COMPLEX:-${REPO_ROOT}/build/kokkos_ep_demo_complex}"

if [[ ! -x "$EXE" ]]; then
  echo "error: not executable: $EXE" >&2
  echo "  Build first, or set KOKKOS_EP_DEMO_COMPLEX to the kokkos_ep_demo_complex path." >&2
  exit 1
fi

"${EXE}" "$@"
