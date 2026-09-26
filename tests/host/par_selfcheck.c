/*
 * Host self-check entry point for the CPU<->CUDA parity comparator.
 *
 * The comparator (tests/cuda/parity_test.c) is dual-buildable: without
 * -DMM_WITH_CUDA it replays the CPU-written golden file against this
 * machine's CPU reference — the oracle pre-gate that verifies the golden
 * reader, observable sizing, walk structure, control-buffer fills, and the
 * comparator itself, everything except the GPU kernels (READ_MEMORY.md
 * §4.5; GPU_VALIDATION.md §5.2).
 *
 * This file exists so the host self-check has an explicit home under
 * tests/host/ and a canonical make target ('make parity-selfcheck'); it
 * adds no logic and shares one source of truth with the GPU build.
 *
 * Build (host, no GPU):  make parity-selfcheck
 * Run:                   build/par_selfcheck build/par_golden.bin
 * Expected:              PARITY PASSED
 */
#include "../cuda/parity_test.c"
