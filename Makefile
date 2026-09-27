# ============================================================================
# mimfer — canonical build system (GNU Make only)
#
# By design there is no CMake/Meson/Bazel or any other framework: one
# Makefile, plain variables, no generated files. All artifacts land in
# build/.
#
# Host builds : gcc, -std=c11 -Wall -Wextra -Werror. Deliberately no
#               optimizer flags: the CPU path is the golden reference and
#               the parity golden is tied to the bit-determinism of this
#               exact recipe (GPU_VALIDATION.md §3.4 "host builds").
# CUDA builds : nvcc, -O2 -fmad=false -DMM_WITH_CUDA -arch=$(CUDA_ARCH).
#
# Quick start:
#   make            build all host targets
#   make test       run the host test suite (plan, rope, flags, device gate,
#                   artifact, engine smoke, engine features, slot lifecycle,
#                   parity self-check)
#   make cuda       build the GPU tests (needs nvcc; default arch sm_120)
#   make help       document every target
# ============================================================================

# ---- tools ------------------------------------------------------------------
CC      ?= gcc
NVCC    ?= nvcc

# The first file target in this file (build/plan_test) is NOT what a bare
# 'make' should build; pin the default goal explicitly.
.DEFAULT_GOAL := all

# Host flags — keep them exact; the golden's bit-determinism depends on them.
CFLAGS  := -std=c11 -Wall -Wextra -Werror
CFLAGS  += -Iinclude -Isrc/model -Isrc/kernels
LDLIBS  := -lm

# GPU build architecture. Default is the reference card (RTX PRO 4000
# Blackwell). Override per machine:  make CUDA_ARCH=sm_90 cuda
CUDA_ARCH ?= sm_120

# GPU flags.
# -fmad=false is MANDATORY until parity validation is complete: the CPU
# reference (the golden oracle) is compiled without FMA contraction;
# allowing FMA on the GPU would change f32 accumulation rounding and break
# the 1-bf16-ulp parity gate on computed ops (GPU_VALIDATION.md §3.4).
# Keep the flag even after parity passes; only drop it if FMA is to be
# enabled and re-validated from scratch.
NVCCFLAGS := -O2 -fmad=false -DMM_WITH_CUDA -arch=$(CUDA_ARCH)
NVCCFLAGS += -Iinclude -Isrc/model -Isrc/kernels
# -lcuda: mm_device_probe() queries the memory clock rate through the driver
# API (cuDeviceGetAttribute), which replaced the cudaDeviceProp field in
# CUDA 13. Link-only flag: it does not change codegen, so the parity-gate
# flags above are untouched. On a machine without a GPU driver, add
# -L$(CUDA toolkit)/lib64/stubs so the linker can resolve it.
NVCCFLAGS += -lcuda
# Extra nvcc flags, appended after NVCCFLAGS (no codegen impact). On a
# machine WITHOUT a GPU driver the linker cannot resolve -lcuda; point it
# at the toolkit's link stub:  make cuda NVCC_EXTRA=-L<cuda toolkit>/lib64/stubs
NVCC_EXTRA ?=

BUILDDIR := build

# ---- sources ------------------------------------------------------------------
# plan_test set (planner-only: no engine, no kernel dispatcher).
PLAN_SRCS := \
    src/plan/plan.c \
    src/alloc/alloc.c \
    src/config/config.c \
    src/core/mimfer.c \
    src/cuda/cuda_rt.c \
    src/sampling/sampling.c

# Full engine set (all engine-linked host binaries).
ENGINE_SRCS := \
    src/engine/engine.c \
    src/plan/plan.c \
    src/alloc/alloc.c \
    src/config/config.c \
    src/core/mimfer.c \
    src/cuda/cuda_rt.c \
    src/cuda/cuda_mem.c \
    src/sampling/sampling.c \
    src/kv/kv.c \
    src/kernels/cpu/cx.c \
    src/kernels/kx.c \
    src/model/tensor_registry.c \
    src/rope/rope.c \
    src/sched/sched.c

# I/O modules: the .mimfer artifact container, the byte-BPE tokenizer and
# the telemetry ring. Linked into the engine set (the engine owns the
# artifact handle, the embedded tokenizer and the telemetry ring) and the
# artifact fixture test below.
IO_SRCS := \
    src/artifact/artifact.c \
    src/tokenizer/tokenizer.c \
    src/telemetry/telemetry.c
ENGINE_SRCS += $(IO_SRCS)

# Small feature sets (unit tests without the engine): RoPE tables and the
# CLI flag surface, plus the config validation and policy checks they call.
CFG_TEST_SRCS := \
    src/config/config.c \
    src/core/mimfer.c \
    src/sampling/sampling.c
ROPE_TEST_SRCS  := src/rope/rope.c $(CFG_TEST_SRCS)
FLAGS_TEST_SRCS := src/flags/flags.c $(CFG_TEST_SRCS)

# GPU set: the same engine sources, with the CPU reference kernels replaced
# by the CUDA launch set.
GPU_SRCS := $(filter-out src/kernels/cpu/cx.c,$(ENGINE_SRCS)) src/kernels/cuda/cx.cu
# nvcc compiles the .c files as C (the dispatcher ABI is pinned to C
# linkage) and cx.cu as CUDA — by EXTENSION. Do NOT pass -x: in CUDA 13.1
# it is a GLOBAL last-value-wins option (nvcc warns "incompatible
# redefinition for option 'x'"), so ANY -x on the line forces one language
# on ALL files; the old per-source "-x c" approach compiled cx.cu as plain
# C (no __global__, no <<<>>>). Verified against the real toolkit, 2026-09-26.
GPU_C_SRCS := $(filter %.c,$(GPU_SRCS))
# The .cu launch set must not be dropped: it defines the kx_op_* launchers
# for the GPU build (replacing cpu/cx.c) plus kx_cuda_oppref. Omitting it
# from the link line is a hard link failure (undefined kx_op_* /
# kx_cuda_oppref).
GPU_CU_SRCS := $(filter %.cu,$(GPU_SRCS))

# Header changes force a rebuild (one-shot builds, no .d tracking).
HDRS := $(wildcard include/mimfer/*.h) src/kernels/kx.h src/model/tensor_registry.h

# ---- artifacts ------------------------------------------------------------------
PLAN_TEST     := $(BUILDDIR)/plan_test
ENGINE_SMOKE  := $(BUILDDIR)/engine_smoke
ROPE_TEST     := $(BUILDDIR)/rope_test
FLAGS_TEST    := $(BUILDDIR)/flags_test
DEVICE_GATE_TEST := $(BUILDDIR)/device_gate_test
ENG_FEATURES  := $(BUILDDIR)/engine_features
PAR_GOLDEN    := $(BUILDDIR)/par_golden
PAR_SELFCHECK := $(BUILDDIR)/par_selfcheck
GOLDEN_BIN    := $(BUILDDIR)/par_golden.bin
PARITY_TEST   := $(BUILDDIR)/parity_test
GPU_SMOKE     := $(BUILDDIR)/gpu_smoke
SLOT_LIFECYCLE := $(BUILDDIR)/slot_lifecycle_test
GPU_SLOT      := $(BUILDDIR)/gpu_slot
ART_TEST      := $(BUILDDIR)/artifact_test

HOST_BINS := $(PLAN_TEST) $(ENGINE_SMOKE) $(ROPE_TEST) $(FLAGS_TEST) \
             $(DEVICE_GATE_TEST) $(ENG_FEATURES) $(SLOT_LIFECYCLE) \
             $(ART_TEST) $(PAR_GOLDEN) \
             $(PAR_SELFCHECK)
CUDA_BINS := $(PARITY_TEST) $(GPU_SMOKE) $(GPU_SLOT)

NVCC_OK := $(shell command -v $(NVCC) >/dev/null 2>&1 && echo yes)

.DELETE_ON_ERROR:

# ---- build rules: host ------------------------------------------------------------
# Each recipe creates its own output directory (mkdir -p $(@D)). A directory
# make-target as order-only prerequisite is a GNU Make 4.3 trap: the first
# run after 'make clean' silently skips the dependent targets.

$(PLAN_TEST): tests/host/plan_test.c $(PLAN_SRCS) $(HDRS)
	mkdir -p $(@D)
	$(CC) $(CFLAGS) tests/host/plan_test.c $(PLAN_SRCS) $(LDLIBS) -o $@

$(ENGINE_SMOKE): tests/host/engine_smoke.c $(ENGINE_SRCS) $(HDRS)
	mkdir -p $(@D)
	$(CC) $(CFLAGS) tests/host/engine_smoke.c $(ENGINE_SRCS) $(LDLIBS) -o $@

$(ROPE_TEST): tests/host/rope_test.c $(ROPE_TEST_SRCS) $(HDRS)
	mkdir -p $(@D)
	$(CC) $(CFLAGS) tests/host/rope_test.c $(ROPE_TEST_SRCS) $(LDLIBS) -o $@

$(FLAGS_TEST): tests/host/flags_test.c $(FLAGS_TEST_SRCS) $(HDRS)
	mkdir -p $(@D)
	$(CC) $(CFLAGS) tests/host/flags_test.c $(FLAGS_TEST_SRCS) $(LDLIBS) -o $@

# Device gate (mm_device_check): name/CC/VRAM/bandwidth policy against the
# target profile; VRAM tolerance regression-anchored to the measured
# driver value on silicon (23.4256 GiB reported on the 24 GB card).
$(DEVICE_GATE_TEST): tests/host/device_gate_test.c $(CFG_TEST_SRCS) $(HDRS)
	mkdir -p $(@D)
	$(CC) $(CFLAGS) tests/host/device_gate_test.c $(CFG_TEST_SRCS) $(LDLIBS) -o $@

# Artifact/tokenizer/telemetry fixture test: in-process .mimfer fixtures
# (written with the modules' own writer), container integrity + section
# verification + corruption cases, the byte-BPE encode/decode, the mbuf
# codec, the telemetry ring, and the engine's real --artifact path
# (open + verify + tokenizer load, determinism, hard-fail negatives).
$(ART_TEST): tests/host/artifact_test.c $(ENGINE_SRCS) $(HDRS)
	mkdir -p $(@D)
	$(CC) $(CFLAGS) tests/host/artifact_test.c $(ENGINE_SRCS) $(LDLIBS) -o $@

# Engine feature gates: YaRN table e2e, spec/vision refusals.
$(ENG_FEATURES): tests/host/engine_features_test.c $(ENGINE_SRCS) $(HDRS)
	mkdir -p $(@D)
	$(CC) $(CFLAGS) tests/host/engine_features_test.c $(ENGINE_SRCS) $(LDLIBS) -o $@

# Slot lifecycle regression (multi-request teardown): sequence A completes
# and its slot is torn down (KV blocks released, linear/conv state reset,
# block-table row zeroed); sequence B then runs on the slot A freed and must
# match a fresh engine's B stream byte-for-byte. One shared source: built
# with gcc here and with nvcc for the GPU binary (GPU_SLOT) below.
$(SLOT_LIFECYCLE): tests/host/slot_lifecycle_test.c $(ENGINE_SRCS) $(HDRS)
	mkdir -p $(@D)
	$(CC) $(CFLAGS) tests/host/slot_lifecycle_test.c $(ENGINE_SRCS) $(LDLIBS) -o $@

$(PAR_GOLDEN): tests/host/par_golden.c $(ENGINE_SRCS) $(HDRS)
	mkdir -p $(@D)
	$(CC) $(CFLAGS) tests/host/par_golden.c $(ENGINE_SRCS) $(LDLIBS) -o $@

# Host self-check = the dual-buildable parity comparator without
# MM_WITH_CUDA (tests/host/par_selfcheck.c is a thin wrapper that
# #includes tests/cuda/parity_test.c — one source of truth).
$(PAR_SELFCHECK): tests/host/par_selfcheck.c $(ENGINE_SRCS) $(HDRS)
	mkdir -p $(@D)
	$(CC) $(CFLAGS) tests/host/par_selfcheck.c $(ENGINE_SRCS) $(LDLIBS) -o $@

# ---- build rules: CUDA ---------------------------------------------------------------
$(PARITY_TEST): tests/cuda/parity_test.c $(GPU_SRCS) $(HDRS)
	mkdir -p $(@D)
	$(NVCC) $(NVCCFLAGS) $(NVCC_EXTRA) tests/cuda/parity_test.c $(GPU_C_SRCS) $(GPU_CU_SRCS) -o $@

$(GPU_SMOKE): tests/cuda/gpu_smoke.c $(GPU_SRCS) $(HDRS)
	mkdir -p $(@D)
	$(NVCC) $(NVCCFLAGS) $(NVCC_EXTRA) tests/cuda/gpu_smoke.c $(GPU_C_SRCS) $(GPU_CU_SRCS) -o $@

# GPU build of the slot lifecycle regression: the SAME source as the host
# binary (tests/host/slot_lifecycle_test.c), compiled by nvcc as C against
# the GPU engine set (see the no-`-x` note above).
$(GPU_SLOT): tests/host/slot_lifecycle_test.c $(GPU_SRCS) $(HDRS)
	mkdir -p $(@D)
	$(NVCC) $(NVCCFLAGS) $(NVCC_EXTRA) tests/host/slot_lifecycle_test.c $(GPU_C_SRCS) $(GPU_CU_SRCS) -o $@
# ---- targets --------------------------------------------------------------------------
.PHONY: all host cuda test tests plan-test engine-smoke rope-test \
        flags-test device-gate-test artifact-test engine-features \
        slot-lifecycle golden \
        parity-selfcheck parity smoke gpu-parity gpu-smoke gpu-slot \
        check-nvcc \
        clean distclean help

all: host

# Default: everything that builds without nvcc.
host: $(HOST_BINS)

check-nvcc:
	@if [ "$(NVCC_OK)" != "yes" ]; then \
	    echo "error: '$(NVCC)' not found on PATH — CUDA builds need the toolkit (>= 12.8)" >&2; \
	    echo "       install it (or point NVCC at it), or use the host path: make host / make test" >&2; \
	    exit 1; \
	fi

# Build both GPU binaries.
cuda: check-nvcc
	$(MAKE) $(CUDA_BINS)
	@echo "GPU binaries built (arch $(CUDA_ARCH)): $(PARITY_TEST) $(GPU_SMOKE) $(GPU_SLOT)"
	@echo "run commands (on a GPU machine): make gpu-parity / make gpu-smoke / make gpu-slot"

# ---- host tests (build + run) ------------------------------------------------------------
plan-test: $(PLAN_TEST)
	@echo "== plan_test (host)"
	$(PLAN_TEST)

engine-smoke: $(ENGINE_SMOKE)
	@echo "== engine_smoke (host)"
	$(ENGINE_SMOKE)

rope-test: $(ROPE_TEST)
	@echo "== rope_test (host)"
	$(ROPE_TEST)

flags-test: $(FLAGS_TEST)
	@echo "== flags_test (host)"
	$(FLAGS_TEST)

device-gate-test: $(DEVICE_GATE_TEST)
	@echo "== device_gate_test (host)"
	$(DEVICE_GATE_TEST)

artifact-test: $(ART_TEST)
	@echo "== artifact_test (host)"
	$(ART_TEST)

engine-features: $(ENG_FEATURES)
	@echo "== engine_features (host)"
	$(ENG_FEATURES)

slot-lifecycle: $(SLOT_LIFECYCLE)
	@echo "== slot_lifecycle_test (host)"
	$(SLOT_LIFECYCLE)

# Golden writer. Phony on purpose: the golden is rewritten on every parity
# self-check, so it is never compared against a stale golden (the golden is
# tied to the CPU reference; kernel changes must re-gate).
golden: $(PAR_GOLDEN)
	$(PAR_GOLDEN) $(GOLDEN_BIN)

# Host parity self-check (oracle pre-gate): expect PARITY PASSED.
parity-selfcheck: $(PAR_SELFCHECK) golden
	@echo "== parity self-check (host oracle gate)"
	$(PAR_SELFCHECK) $(GOLDEN_BIN)

# Short aliases (host side).
parity: parity-selfcheck
smoke:  engine-smoke

# Full host test suite, strictly sequential, fail-fast (recursive make keeps
# the order under -j as well). Unit tests first, then the engine smoke, then
# the parity oracle gate (last, since the golden is regenerated in place).
test:
	$(MAKE) plan-test
	$(MAKE) rope-test
	$(MAKE) flags-test
	$(MAKE) device-gate-test
	$(MAKE) artifact-test
	$(MAKE) engine-smoke
	$(MAKE) engine-features
	$(MAKE) slot-lifecycle
	$(MAKE) parity-selfcheck
	@echo "HOST TEST SUITE PASSED (plan_test, rope_test, flags_test, device_gate_test, artifact_test, engine_smoke, engine_features, slot_lifecycle_test, parity self-check)"

tests: test

# ---- GPU tests (build only; print the run command, never run) ----------------------------
gpu-parity: check-nvcc
	$(MAKE) $(PARITY_TEST)
	@echo "built $(PARITY_TEST) (arch $(CUDA_ARCH)) — not run automatically."
	@echo "on the GPU machine (GPU_VALIDATION.md §6):"
	@echo "  $(PARITY_TEST) $(GOLDEN_BIN)"
	@echo "  (golden first: make golden)"

gpu-smoke: check-nvcc
	$(MAKE) $(GPU_SMOKE)
	@echo "built $(GPU_SMOKE) (arch $(CUDA_ARCH)) — not run automatically."
	@echo "on the GPU machine (GPU_VALIDATION.md §7):"
	@echo "  $(GPU_SMOKE)"

gpu-slot: check-nvcc
	$(MAKE) $(GPU_SLOT)
	@echo "built $(GPU_SLOT) (arch $(CUDA_ARCH)) — not run automatically."
	@echo "on the GPU machine (GPU_VALIDATION.md §8):"
	@echo "  $(GPU_SLOT)"

# ---- housekeeping --------------------------------------------------------------------------
clean:
	rm -rf $(BUILDDIR)

distclean: clean
	@echo "distclean complete (build/ removed; nothing else is generated)"

help:
	@echo "mimfer — canonical build targets (GNU Make, artifacts in $(BUILDDIR)/)"
	@echo ""
	@echo "build:"
	@echo "  make              default: build all host targets (same as 'make host')"
	@echo "  make host         gcc, C11, -Wall -Wextra -Werror →"
	@echo "                    $(HOST_BINS)"
	@echo "  make cuda         nvcc, -O2 -fmad=false -DMM_WITH_CUDA -arch=\$$(CUDA_ARCH) →"
	@echo "                    $(CUDA_BINS)   (needs nvcc)"
	@echo ""
	@echo "host tests (build + run, strictly sequential, fail-fast):"
	@echo "  make test         plan-test → rope-test → flags-test → device-gate-test"
	@echo "                    → artifact-test → engine-smoke → engine-features"
	@echo "                    → slot-lifecycle → parity-selfcheck (alias: make tests)"
	@echo "  make plan-test        $(PLAN_TEST)   → PLAN TEST PASSED"
	@echo "  make rope-test        $(ROPE_TEST)   → ROPE TEST PASSED"
	@echo "  make flags-test       $(FLAGS_TEST)   → FLAGS TEST PASSED"
	@echo "  make device-gate-test $(DEVICE_GATE_TEST) → DEVICE GATE TEST PASSED"
	@echo "  make artifact-test    $(ART_TEST)   → ARTIFACT TEST PASSED"
	@echo "  make engine-smoke     $(ENGINE_SMOKE)   → ENGINE SMOKE PASSED"
	@echo "  make engine-features  $(ENG_FEATURES)   → ENGINE FEATURES PASSED"
	@echo "  make slot-lifecycle   $(SLOT_LIFECYCLE)   → SLOT LIFECYCLE PASSED"
	@echo "  make golden           $(GOLDEN_BIN)  → PAR GOLDEN WRITTEN"
	@echo "  make parity-selfcheck $(PAR_SELFCHECK) → PARITY PASSED"
	@echo "  make parity           alias: make parity-selfcheck"
	@echo "  make smoke            alias: make engine-smoke"
	@echo ""
	@echo "gpu tests (build only; print the run command, never run automatically):"
	@echo "  make gpu-parity     $(PARITY_TEST)   (run with the golden: '$(PARITY_TEST) $(GOLDEN_BIN)')"
	@echo "  make gpu-smoke      $(GPU_SMOKE)"
	@echo "  make gpu-slot       $(GPU_SLOT)      (slot lifecycle regression)"
	@echo ""
	@echo "housekeeping:"
	@echo "  make clean          remove $(BUILDDIR)/"
	@echo "  make distclean      same as clean (nothing else is generated)"
	@echo "  make help           this message"
	@echo ""
	@echo "variables:"
	@echo "  CUDA_ARCH ?= sm_120   GPU arch (override: make CUDA_ARCH=sm_90 cuda)"
	@echo "  CC ?= gcc             host compiler"
	@echo "  NVCC ?= nvcc          CUDA compiler"
	@echo "  NVCC_EXTRA ?=         extra nvcc flags (driverless: -L<toolkit>/lib64/stubs)"
	@echo ""
	@echo "note: GPU builds use -fmad=false until parity validation is complete —"
	@echo "the CPU oracle is FMA-free (see the NVCCFLAGS comment, GPU_VALIDATION.md §3.4)."

