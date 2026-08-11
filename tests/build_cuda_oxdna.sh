#!/usr/bin/env bash
#
# Rebuilds oxDNA + oxpy + the rovigatti contrib plugins with CUDA support enabled, so that
# test_energy_conservation.py's CUDA test runs instead of skipping.
#
# This repo's default oxDNA/build is CPU-only -- CUDA was never turned on there (see
# /root/CLAUDE.md, "No GPU/CUDA support baked in yet"). Run this manually, once, on a machine
# that actually has the CUDA toolkit and a visible GPU; it is NOT invoked automatically by the
# test suite, because it reinstalls the `oxpy` package into the active Python/micromamba env
# (replacing the CPU-only one) and can take several minutes.
#
# Usage: bash tests/build_cuda_oxdna.sh [path/to/oxDNA]
# (defaults to the sibling oxDNA checkout, ../../oxDNA relative to this script)
set -euo pipefail

OXDNA_ROOT="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../oxDNA" && pwd)}"
BUILD_DIR="$OXDNA_ROOT/build_cuda"

if ! command -v nvcc >/dev/null 2>&1; then
    echo "nvcc not found on PATH -- install the CUDA toolkit before running this script." >&2
    exit 1
fi

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"
cmake .. -DPYTHON=ON -DCUDA=ON -DCUDA_DOUBLE=ON
# rebuilds oxDNA and reinstalls oxpy (with CUDA support) into the active Python env
make install -j"$(nproc)"
# (re)builds contrib/rovigatti/*.so in place, including the new CUDADetailedPatchySwapInteraction.so
make rovigatti -j"$(nproc)"

echo "Done: oxpy was reinstalled with CUDA support, and $OXDNA_ROOT/contrib/rovigatti now has the CUDA plugin variants."
