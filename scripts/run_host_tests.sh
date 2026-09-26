#!/bin/sh
# ============================================================================
# mimfer — host test suite runner (invoked by CMake's 'test' target)
#
# Mirrors the root Makefile's 'make test' exactly: strictly sequential and
# fail-fast; unit tests first, then the engine smoke tests, then the parity
# oracle gate last (the golden is regenerated in place before the
# self-check, so the gate is never compared against a stale golden).
#
# Usage:
#   sh scripts/run_host_tests.sh <builddir>
#
# <builddir> is the directory holding the host binaries (plan_test,
# rope_test, flags_test, engine_smoke, engine_features, par_golden,
# par_selfcheck). With the CMake build it is build/ (CMAKE_BINARY_DIR);
# with the Makefile build the same layout holds.
# ============================================================================

set -e

BUILDDIR=${1:?usage: run_host_tests.sh <builddir>}

echo "== plan_test (host)"
"$BUILDDIR/plan_test"

echo "== rope_test (host)"
"$BUILDDIR/rope_test"

echo "== flags_test (host)"
"$BUILDDIR/flags_test"

echo "== engine_smoke (host)"
"$BUILDDIR/engine_smoke"

echo "== engine_features (host)"
"$BUILDDIR/engine_features"

# Golden writer — silent, like 'make golden': par_golden prints
# 'PAR GOLDEN WRITTEN ...' itself.
"$BUILDDIR/par_golden" "$BUILDDIR/par_golden.bin"

echo "== parity self-check (host oracle gate)"
"$BUILDDIR/par_selfcheck" "$BUILDDIR/par_golden.bin"

echo "HOST TEST SUITE PASSED (plan_test, rope_test, flags_test, engine_smoke, engine_features, parity self-check)"
