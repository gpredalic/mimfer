# READ_MEMORY.md — mimfer

> **Purpose:** Single source of truth for project state. Read this first before
> touching any code. Update it immediately whenever project state materially
> changes.
>
> **Last updated:** 2026-09-26 (session handoff: CUDA 13.1 real-toolkit
> build — the GPU build now COMPILES + LINKS with the real local
> nvcc 13.1.115, driverless via the toolkit link stubs; root cause of
> the old recipe failure found and fixed: nvcc 13.1's `-x` is a GLOBAL
> last-value-wins option, so per-source `-x c` compiled `cx.cu` as plain
> C — all `-x` removed, language by extension (Makefile + CMake);
> `extern "C"` guards added to 17 public headers (C++ TU linkage);
> `cx.cu` C-only idioms fixed (comma `dim3` decls, `malloc` casts,
> `math.h`); `config.c` CUDA 13.1 driver-API probe; `-lcuda` in both
> link lines; both build systems verified end-to-end, host suite
> re-green, fatbin inspected (14 kernels, sm_120, zero warnings);
> GPU_VALIDATION.md / RELEASE_READINESS.md / this memory synced)
> **Project health:** GREEN
> **Host correctness:** VERIFIED · **Memory safety:** VERIFIED · **Determinism:** VERIFIED
> **GPU build:** VERIFIED (real nvcc 13.1.115, sm_120 fatbin inspected, zero warnings) · **GPU execution:** PENDING (no GPU driver on this host)
> **External validation pack:** COMPLETE (GPU_VALIDATION.md runbook, VALIDATION_CHECKLIST.md go/no-go, RELEASE_READINESS.md status/risks)
> **Build system:** COMPLETE — Makefile (canonical) + CMakeLists.txt (parity, optional CUDA via `-DMIMFER_ENABLE_CUDA=ON`); both verified end-to-end 2026-09-26

---

## 1. Current Status Board

CPU-reference execution path is **COMPLETE and VERIFIED**.

| Status | Component |
|--------|-----------|
| [COMPLETE] | Planner |
| [COMPLETE] | Host Planner Validation |
| [COMPLETE] | Kernel Dispatcher |
| [COMPLETE] | CPU Reference Kernels |
| [COMPLETE] | Tensor Registry |
| [COMPLETE] | Engine Skeleton |
| [COMPLETE] | Engine Smoke Test |
| [COMPLETE] | KV Block Allocation Validation |
| [COMPLETE] | CUDA Runtime Integration |
| [COMPLETE] | CUDA Kernels (written, parity-first; pending GPU verification) |
| [COMPLETE] | CPU/CUDA parity tooling (golden writer + parity test + self-check) |
| [COMPLETE] | RoPE frequency tables (plain + YaRN NTK-by-parts) — `src/rope/rope.c` |
| [COMPLETE] | CLI flag surface (one grammar table + cross-field validation) — `src/flags/flags.c` |
| [COMPLETE] | Weights profiles (quasar / neroued context defaults + artifact metadata) |
| [COMPLETE] | Engine feature gates: `--spec` refusal, `--vision` hook, `--weights-profile` |
| [COMPLETE] | Engine feature test harness (`engine_features_test.c`: YaRN e2e + refusals) |
| [COMPLETE] | End-to-end GPU smoke test (written; pending GPU execution) |
| [COMPLETE] | External validation pack: GPU_VALIDATION.md (runbook), VALIDATION_CHECKLIST.md (go/no-go), RELEASE_READINESS.md (status/risks/audit) |
| [COMPLETE] | README.md (project README: current model support, artifact support, hardware scope, extensibility) |
| [COMPLETE] | Project governance: .clinerules (mandatory rules), .gitignore, git repo initialized (main) |
| [COMPLETE] | Canonical build system: root Makefile (GNU Make only) — host + CUDA build rules, `make` / `make test` / `make cuda` / `make help` / `make clean` / `make distclean`; artifacts in `build/` |
| [COMPLETE] | Final build audit (2026-09-26): `BUILD_AUDIT.md` (findings B1–B10, conformance matrix, evidence), `SOURCE_TREE.md` (annotated file inventory), `MODULE_DEPENDENCIES.md` (include graph + module map); two critical GPU-link defects fixed (B1, B10) |
| [PENDING]  | GPU execution of parity + smoke tests (GPU build now compiles + links driverless with real nvcc 13.1; a GPU host with the driver is what's missing) |
| [PENDING]  | Blackwell Optimizations |

Note: the CUDA execution path is written in full — `src/cuda/cuda_rt.c`
(streams, events, graph capture, error-checked API wrappers, honest no-op
host stubs), `src/cuda/cuda_mem.c` (device arenas + pinned I/O block), and
`src/kernels/cuda/cx.cu` (the full GPU kernel launch set, parity-first
mirrors of the CPU reference, built with `-fmad=false`). What is PENDING is
**running it**: this host has no `nvcc`, CUDA toolkit, or GPU, so the two
GPU tests (§4.6) are written and syntax-checked but must be built and run on
a GPU machine. `tests/cuda/parity_test.c` additionally builds on the host
as a self-check (§4.5) and PASSES there, which verifies everything except
the GPU kernels themselves.

---

## 2. Current Execution Flow (FULL HOST PATH WORKING)

```
load model
  → register tensors
  → build plan
  → capture plan
  → execute planner ops
  → kernel dispatcher (src/kernels/kx.c)
  → CPU reference kernels (src/kernels/cpu/cx.c)
  → sampling
  → next token
```

---

## 3. Architecture Notes (STABLE — do not change)

- **Planner remains isolated.** Tensor registry remains **outside** the planner.
- **Engine owns:** tensor registration, model loading, runtime orchestration.
- **Planner owns:** execution graph generation, execution planning, activation planning.
- **Do NOT move responsibilities between modules.**
- **Do NOT rewrite the planner. Do NOT redesign the engine. Do NOT introduce a
  generic backend abstraction.**
- The CUDA path must **attach to the existing validated host architecture**.

### Source layout

```
include/mimfer/   public headers (engine.h, plan.h, kernels.h, kv.h, cuda_rt.h,
                  tensor.h, sched.h, sampling.h, config.h, alloc.h, artifact.h,
                  telemetry.h, tokenizer.h, mimfer.h, rope.h, flags.h)
src/engine/       engine.c          — runtime orchestration (create/load/step;
                                      spec-backend + vision feature gates)
src/plan/         plan.c            — planner (isolated; owns op-graph + activation planning)
src/kernels/      kx.c, kx.h        — kernel dispatcher (mm_kx_invoke)
src/kernels/cpu/  cx.c              — CPU reference kernels (RoPE consumes the
                                      engine's frequency table via the control buffer)
src/model/        tensor_registry.c/.h — tensor registry (NOT part of planner)
src/kv/           kv.c              — KV block pool (block IDs are 1..n_blocks)
src/sched/        sched.c           — request scheduler (submit/active/queued)
src/sampling/     sampling.c        — sampling
src/alloc/        alloc.c           — arena allocator (mm_arena_init_host, ...)
src/config/       config.c          — model cfg finalize, engine cfg validation
                                      (cross-field flag rules), weights profiles
                                      (MM_PROFILES: quasar / neroued)
src/rope/         rope.c            — RoPE frequency tables: plain + YaRN
                                      (NTK-by-parts); plain is bit-identical to
                                      the legacy per-kernel formula (golden-safe)
src/flags/        flags.c           — CLI flag grammar (one table; --help;
                                      explicit errors; no unknown-flag fallback)
src/core/         mimfer.c          — core utilities
src/cuda/         cuda_rt.c         — CUDA runtime layer (MM_WITH_CUDA guarded; host no-op stubs)
                  cuda_mem.c        — device arenas + pinned host I/O block (MM_WITH_CUDA)
src/kernels/cuda/ cx.cu             — GPU kernel launch set (nvcc, -DMM_WITH_CUDA, -fmad=false)
src/artifact/     artifact.c        — .mimfer artifact loading (written; NOT integrated — see RELEASE_READINESS.md §3)
src/tokenizer/    tokenizer.c       — written; NOT integrated (same status)
src/telemetry/    telemetry.c       — written; NOT integrated (same status)
Makefile              canonical GNU Make build system (host + CUDA, artifacts in build/)
docs/             dflash2.md        — declared scope of the speculative-decoding
                                      surface (planned; scaffolding only)
tests/host/       plan_test.c, engine_smoke.c, rope_test.c (RoPE tables vs reference
                  formula), flags_test.c (full CLI grammar + cross-field rules),
                  engine_features_test.c (e2e YaRN table reach + spec/vision
                  refusals), par_golden.c (golden writer), par_selfcheck.c (thin
                  wrapper: #includes tests/cuda/parity_test.c, one shared
                  comparator for the host self-check)
tests/cuda/       parity_test.c (GPU comparator; host self-check build via the
                  wrapper above), gpu_smoke.c (end-to-end GPU smoke test)
```

---
## 4. Verified Build Commands (KNOWN-GOOD)

The commands below are re-verified from the project root
(`/home/razvijalec/mimfer`): clean compile under `-Werror`, tests pass.
**Every engine-linked build (host or CUDA) includes `src/rope/rope.c`** —
`src/engine/engine.c` calls `mm_rope_init`/`mm_rope_fill_ctrl`, so a link
set without the RoPE module fails with undefined references. The canonical
way to build is the root `Makefile` (§4.7); the manual commands are the
reference recipes with identical flags.

### 4.1 plan_test — KNOWN-GOOD BUILD

```sh
gcc -std=c11 -Wall -Wextra -Werror -Iinclude \
    tests/host/plan_test.c src/plan/plan.c src/alloc/alloc.c \
    src/config/config.c src/core/mimfer.c src/cuda/cuda_rt.c \
    src/sampling/sampling.c -o /tmp/plan_test
/tmp/plan_test
```

- The test stubs `mm_kx_invoke` and `mm_tens_find` locally; it is self-contained.
- Expected output: `PLAN TEST PASSED (prefill 92 ops, decode 92 ops)`

### 4.2 engine_smoke — KNOWN-GOOD BUILD

```sh
gcc -std=c11 -Wall -Wextra -Werror -Iinclude -Isrc/model -Isrc/kernels \
    tests/host/engine_smoke.c \
    src/engine/engine.c src/plan/plan.c src/alloc/alloc.c \
    src/config/config.c src/core/mimfer.c src/cuda/cuda_rt.c \
    src/sampling/sampling.c src/kv/kv.c src/kernels/cpu/cx.c \
    src/kernels/kx.c src/model/tensor_registry.c src/rope/rope.c \
    src/sched/sched.c \
    -lm -o /tmp/engine_smoke
/tmp/engine_smoke
```

- **The two extra include dirs are mandatory:** `-Isrc/model` (for
  `tensor_registry.h`, included by `src/engine/engine.c`) and `-Isrc/kernels`
  (for `kx.h`, included by `src/kernels/cpu/cx.c`). A build without them fails
  with `fatal error: tensor_registry.h: No such file or directory`.
- Expected output:
  ```
    short (block 0)        rounds=17 tokens_out=17 ...
    long (crosses block)   rounds=71 tokens_out=71 ...
  ENGINE SMOKE PASSED
  ```

### 4.3 Sanitizer builds (verification builds)

```sh
# engine_smoke under ASan + UBSan (LeakSan is on by default)
gcc -std=c11 -Wall -Wextra -Werror -Iinclude -Isrc/model -Isrc/kernels \
    -fsanitize=address,undefined -fno-sanitize-recover=all -g \
    tests/host/engine_smoke.c \
    src/engine/engine.c src/plan/plan.c src/alloc/alloc.c \
    src/config/config.c src/core/mimfer.c src/cuda/cuda_rt.c \
    src/sampling/sampling.c src/kv/kv.c src/kernels/cpu/cx.c \
    src/kernels/kx.c src/model/tensor_registry.c src/rope/rope.c \
    src/sched/sched.c \
    -lm -o /tmp/engine_smoke_asan
/tmp/engine_smoke_asan   # → clean, exit 0

# plan_test under ASan + UBSan, leaks disabled (see §5 caveat)
gcc -std=c11 -Wall -Wextra -Werror -Iinclude \
    -fsanitize=address,undefined -fno-sanitize-recover=all -g \
    tests/host/plan_test.c src/plan/plan.c src/alloc/alloc.c \
    src/config/config.c src/core/mimfer.c src/cuda/cuda_rt.c \
    src/sampling/sampling.c -o /tmp/plan_test_asan
ASAN_OPTIONS=detect_leaks=0 /tmp/plan_test_asan   # → functional PASS, exit 0
```

### 4.4 par_golden — golden writer (KNOWN-GOOD BUILD, VERIFIED)

```sh
gcc -std=c11 -Wall -Wextra -Werror -Iinclude -Isrc/model -Isrc/kernels \
    tests/host/par_golden.c src/engine/engine.c src/plan/plan.c \
    src/alloc/alloc.c src/config/config.c src/core/mimfer.c \
    src/cuda/cuda_rt.c src/cuda/cuda_mem.c src/sampling/sampling.c \
    src/kv/kv.c src/kernels/cpu/cx.c src/kernels/kx.c \
    src/model/tensor_registry.c src/rope/rope.c src/sched/sched.c -lm \
    -o /tmp/par_golden
/tmp/par_golden /tmp/par_golden.bin
```

- Runs the CPU reference op by op (1 prefill + 16 decode rounds, seed 12345,
  prompt `{5,9,17,42}`, chunk 8, greedy) and records every observable into
  `/tmp/par_golden.bin` (magic `MMPAR001`, ~3.8 MB, little-endian).
- Expected output:
  ```
    coverage: CPU reference covers 18/18 executable opcodes
  PAR GOLDEN WRITTEN /tmp/par_golden.bin (1 prefill + 16 decode rounds, seed 12345)
  ```
- **Golden file format** (the parity test consumes this; see
  `tests/host/par_golden.c` header for the authoritative spec):
  header `MMPAR001 | seed max_ctx max_out chunk n_rounds`; per round:
  `round_id M n_ops`; per op: `op_type layer n0 n1 n2 n_obs` then per
  observable `tag nbytes bytes[tag]` (tag 0..3 = operand a..d; 4 = KV pool,
  5 = lin f32 state, 6 = conv bf16 tail — the pool/state trio follows the
  last op of every round); terminator `u32 0`. Tolerance classes are NOT
  stored; the parity test derives them from (op, tag): class 0 bit-exact
  (EMBED/COPY/SAMPLE, KV pool), class 1 1 bf16 ulp (computed bf16), class 2
  4 f32 ulp / 1 bf16 ulp (recurrence + conv state).
- The golden is written by the CPU oracle only; it is input data for the GPU
  test, never compared against itself on a GPU machine.

### 4.5 par_selfcheck — parity test, host self-check (KNOWN-GOOD BUILD, VERIFIED)

```sh
gcc -std=c11 -Wall -Wextra -Werror -Iinclude -Isrc/model -Isrc/kernels \
    tests/cuda/parity_test.c src/engine/engine.c src/plan/plan.c \
    src/alloc/alloc.c src/config/config.c src/core/mimfer.c \
    src/cuda/cuda_rt.c src/cuda/cuda_mem.c src/sampling/sampling.c \
    src/kv/kv.c src/kernels/cpu/cx.c src/kernels/kx.c \
    src/model/tensor_registry.c src/rope/rope.c src/sched/sched.c -lm \
    -o /tmp/par_selfcheck
/tmp/par_selfcheck /tmp/par_golden.bin
```

- `tests/cuda/parity_test.c` is **dual-buildable**: without
  `-DMM_WITH_CUDA` it replays the golden file against the CPU reference
  (d2h() = memcpy, device I/O no-ops). It verifies the golden reader, the
  observable sizing, the walk structure and the comparator end to end —
  everything except the GPU kernels themselves.
- Expected output (re-verified 2026-09-25):
  ```
    coverage: CPU reference covers 18/18 executable opcodes
    compared 1564 ops, 2108 observables, 2009424 elements (2009424 bit-exact)
    max deviation class 1 (computed bf16): 0 f32 ulp = 0.000 bf16 ulp (tol 1 bf16 ulp)
    max deviation class 2 (f32 state / bf16 conv tail): 0 f32 ulp (tol 4 f32 / 1 bf16 ulp)
  PARITY PASSED (/tmp/par_golden.bin: 1 prefill + 16 decode rounds)
  ```

### 4.6 CUDA builds — BUILD-VERIFIED on the real toolkit, execution PENDING

This host has the **real CUDA 13.1 toolkit** (nvcc 13.1.115) but no GPU
driver. The GPU build (`make cuda` / CMake `-DMIMFER_ENABLE_CUDA=ON`)
COMPILES and LINKS cleanly here via the toolkit's driver link stubs
(`NVCC_EXTRA` / `MIMFER_NVCC_EXTRA` = `-L<toolkit>/lib64/stubs`;
`-lcuda` is in the link line). Verified 2026-09-26: zero warnings, sm_120
fatbin inspected (all 14 kernels present). What remains is execution on a
GPU with the driver.

**nvcc `-x` root cause (fixed):** CUDA 13.1's `-x` is a *global*
last-value-wins option (nvcc warns "incompatible redefinition for option
'x'"), so the old per-source `-x c` flags forced plain C on **every**
file, including `cx.cu`. All `-x` flags are gone from the Makefile, CMake,
and documented recipes — nvcc now picks the language by extension (`.c` →
C, `.cu` → CUDA/C++).

**Canonical runbook: `GPU_VALIDATION.md`** (machine/CUDA/compiler
requirements, the full ordered procedure incl. the `par_selfcheck` oracle
pre-gate, expected outputs, troubleshooting, bug-report template). The
commands below are **verbatim** from its §6.1/§7.1 (Makefile recipes are
identical). Go/no-go gate: `VALIDATION_CHECKLIST.md`.

```sh
# Parity test (numeric gate): compares GPU op outputs to the CPU golden.
# Run on the GPU machine after building /tmp/par_golden.bin on ANY host
# (the golden is CPU-written and machine-independent):
nvcc -O2 -fmad=false -DMM_WITH_CUDA -arch=sm_120 \
     -Iinclude -Isrc/model -Isrc/kernels -lcuda \
     tests/cuda/parity_test.c \
     src/engine/engine.c src/plan/plan.c \
     src/alloc/alloc.c src/config/config.c \
     src/core/mimfer.c src/cuda/cuda_rt.c \
     src/cuda/cuda_mem.c src/sampling/sampling.c \
     src/kv/kv.c src/kernels/kx.c \
     src/model/tensor_registry.c src/rope/rope.c \
     src/sched/sched.c \
     src/kernels/cuda/cx.cu \
     -o /tmp/parity_test
/tmp/parity_test /tmp/par_golden.bin

# End-to-end GPU smoke test (graph + direct dispatch, oracle heads):
nvcc -O2 -fmad=false -DMM_WITH_CUDA -arch=sm_120 \
     -Iinclude -Isrc/model -Isrc/kernels -lcuda \
     tests/cuda/gpu_smoke.c \
     src/engine/engine.c src/plan/plan.c \
     src/alloc/alloc.c src/config/config.c \
     src/core/mimfer.c src/cuda/cuda_rt.c \
     src/cuda/cuda_mem.c src/sampling/sampling.c \
     src/kv/kv.c src/kernels/kx.c \
     src/model/tensor_registry.c src/rope/rope.c \
     src/sched/sched.c \
     src/kernels/cuda/cx.cu \
     -o /tmp/gpu_smoke
/tmp/gpu_smoke
```

- **`-fmad=false` is mandatory** for `cx.cu`: the CPU reference (baseline
  x86-64) emits mul+add only; allowing FMA contraction on the GPU would
  change f32 accumulation rounding and blow the 1-ulp bf16 parity gate.
- Linkage note: the 17 public headers pin C linkage on the dispatcher ABI
  (`extern "C"` guards) because the launchers are DEFINED in the C++
  translation unit `cx.cu` while the dispatcher and tests are C. Do not
  remove those guards.
- Deeper GPU verification once it runs: `compute-sanitizer` (or
  `cuda-memcheck`) around `/tmp/gpu_smoke`.
- Expected outputs (oracle-derived; the smoke test hardcodes the first 8
  tokens of the host CPU reference and checks them):
  ```
  # /tmp/parity_test  →  PARITY PASSED (/tmp/par_golden.bin: 1 prefill + 16 decode rounds)
  # /tmp/gpu_smoke    →
    short (block 0)        mode=graph+direct rounds=17 tokens_out=17 head=[451 20 430 317 0 152 24 414]
    long (crosses block)   mode=graph+direct rounds=71 tokens_out=71 head=[127 230 247 387 485 327 294 387]
  GPU SMOKE PASSED
  ```

### 4.7 Canonical builds: the root `Makefile` (GNU Make) + `CMakeLists.txt` (parity; VERIFIED 2026-09-26)

The `Makefile` at the repository root is the **canonical** build/validate
system; `CMakeLists.txt` (CMake 3.16+, Ninja) is a parity build producing the
same 7 host binaries with identical flags, plus optional CUDA
(`-DMIMFER_ENABLE_CUDA=ON`, `-DMIMFER_NVCC_EXTRA` for driverless stub
linking; default target stays host-only, `--target cuda` for the GPU
binaries — mirroring `make` default / `make cuda`). Plain variables, no
generated files. All artifacts land in `build/` (Makefile) / `build/`
(CMake) — git-ignored; the Makefile, CMakeLists.txt and `tests/` are
explicitly NOT ignored.

| Target | Effect |
|--------|--------|
| `make` (default) | build all host targets (same as `make host`) → `build/plan_test`, `build/engine_smoke`, `build/rope_test`, `build/flags_test`, `build/engine_features`, `build/par_golden`, `build/par_selfcheck` |
| `make host` | same as default |
| `make test` / `make tests` | run the host suite strictly sequentially (fail-fast): `plan-test` → `rope-test` → `flags-test` → `engine-smoke` → `engine-features` → `parity-selfcheck` (the last builds `par_golden`, writes `build/par_golden.bin`, then replays it) — unit tests first, engine smoke, feature gates, parity oracle gate last |
| `make plan-test` / `make rope-test` / `make flags-test` / `make engine-smoke` / `make engine-features` / `make golden` / `make parity-selfcheck` | single test targets (aliases: `make parity` → parity-selfcheck, `make smoke` → engine-smoke) |
| `make cuda` | build `build/parity_test` + `build/gpu_smoke` with nvcc (needs nvcc on PATH; prints a run command, never runs GPU tests; driverless hosts: `make cuda NVCC_EXTRA=-L<toolkit>/lib64/stubs`) |
| `make gpu-parity` / `make gpu-smoke` | CUDA build-only, with the run command printed |
| `make check-nvcc` | nvcc presence probe (polite error if missing) |
| `make help` | document every target |
| `make clean` / `make distclean` | remove `build/` |

- **Host flags are fixed by design** (`-std=c11 -Wall -Wextra -Werror`, no
  optimizer flags): the CPU path is the golden reference and the golden's
  bit-determinism is tied to this exact recipe (GPU_VALIDATION.md §3.4).
- **GPU flags:** `nvcc -O2 -fmad=false -DMM_WITH_CUDA -arch=$(CUDA_ARCH)`,
  `CUDA_ARCH ?= sm_120` (RTX PRO 4000 Blackwell). `-fmad=false` stays
  mandatory until parity validation is complete (see §4.6 note).
- `tests/host/par_selfcheck.c` is a **thin wrapper** that `#include`s
  `tests/cuda/parity_test.c` (compiled without `-DMM_WITH_CUDA`) — one
  source of truth for the comparator, no duplicated parity logic.
- **Two GNU Make 4.3 traps found and fixed while building this (do not
  reintroduce):**
  1. A directory make-target used as an *order-only prerequisite*
     (`target: prqs | $(BUILDDIR)` + `$(BUILDDIR): mkdir -p $@`): on the
     first run after `make clean` (directory absent) make remakes the
     directory prerequisite and then **silently skips the dependent file
     targets** ("No need to remake target … is up to date" although the
     file does not exist). Fix: each file recipe runs `mkdir -p $(@D)`
     itself; no directory make-target exists.
  2. The **default goal** was the first ordinary target in the file
     (`build/plan_test`), so bare `make` built only that one binary. Fix:
     `.DEFAULT_GOAL := all` pinned near the top of the Makefile.
- Verified end-to-end on 2026-09-26 from a clean tree: `make` builds all
  seven host binaries; `make test` → `PLAN TEST PASSED` + `ROPE TEST
  PASSED` + `FLAGS TEST PASSED` + `ENGINE SMOKE PASSED` + `ENGINE FEATURES
  PASSED` + `PAR GOLDEN WRITTEN build/par_golden.bin` + `PARITY PASSED`
  (2009424 elements bit-exact) + `HOST TEST SUITE PASSED`; repeat `make`
  is a no-op; `make cuda` without nvcc fails with an actionable message.
- The manual `gcc`/`nvcc` commands in §4.1–§4.6 remain valid reference
  builds (identical flags); the Makefile is what you run.

---
## 5. Verified Tests (re-verified 2026-09-25; feature surface re-verified 2026-09-26)

### plan_test — PASS

- prefill: 92 ops, decode: 92 ops.
- Invariants verified:
  - first op is `EMBED`
  - last op is `SAMPLE`
  - correct LIN layer behavior (one `LIN_PRE`/`LIN_DEC` per lin layer)
  - correct FULL layer behavior (one `ATT_PREFILL` per full layer prefill;
    `ATT_DECODE` only, no `ATT_PREFILL`, in decode)
  - prefill and decode emit the same op count
  - deterministic/idempotent planner output (rebuild produces identical plan)

### engine_smoke — PASS

- Short run: **17 rounds** (stays within KV block 0)
- Long run: **71 rounds** (crosses the 64-token KV boundary → block 1)
- Verified:
  - deterministic execution (byte-identical token streams across two runs, same seed)
  - KV block rollover
  - multi-block path (per-slot block table beyond index 0)
  - tokens in valid range (vocab 512)
  - repeated runs identical

### rope_test — PASS (RoPE frequency tables)

- Plain tables are **bit-identical to the legacy per-kernel formula** for the
  tiny shape (θ 1e7, dim 16) — this is what keeps the parity golden unchanged
  when the RoPE kernels moved to the engine-owned table.
- YaRN tables checked against the reference NTK-by-parts formula
  (transformers v4.45.0) for a range of scale factors / original contexts;
  the attention scale is `1 / sqrt(1 + ln(s)/2)` only when `s > 1`.
- Unit-level only (no engine); the e2e reach of the table into the kernels is
  proven by `engine_features_test` below.

### flags_test — PASS (CLI flag surface)

- Every flag in the grammar parses and reaches `mm_engine_cfg` (artifact,
  context, KV, sampling, scheduling, rope, spec, vision, profile).
- Malformed values (bad dtype names, out-of-range ints, `--rope-yarn-factor`
  ≤ 1, `--draft-tokens` 0/9, `--spec` without `--draft-tokens`,
  `--lm-head-draft` without `--spec`) are rejected with explicit errors —
  the parser never silently falls back.
- Cross-field rules via `mm_engine_cfg_validate`; `--weights-profile
  quasar|neroued` lookup and unknown-profile rejection.
- `--help` lists the full surface; `--version` prints the build string.

### engine_features_test — PASS (engine feature gates, host)

- **YaRN e2e reach:** one prefill round with a plain engine and one with a
  YaRN engine (`rope_factor 2.0`) — the post-prefill `q` buffers **differ**
  (the table demonstrably reaches the RoPE kernels in the forward pass),
  while two plain engines are **byte-identical** (the golden path is
  untouched). The buffers are compared *before* the plain engine is
  destroyed (a previous version of this test used the plain engine's `q`
  pointer after destroy — use-after-free; fixed).
- **Spec gate:** `--spec mtp` and `--spec dflash2` are refused at
  `mm_engine_load` with `MM_ERR_UNSUPPORTED` + an explicit "start with
  --spec off" log; `--spec off` boots cleanly.
- **Vision gate:** with `--vision`, image submission validates then returns
  an explicit `MM_ERR_UNSUPPORTED` (hook only; no pipeline).
- The 512-vocab toy model does not guarantee a sampled-token flip within 17
  rounds when RoPE changes, so the e2e assertion is on the rotated `q`
  buffer, not on the token stream.

### par_golden — PASS (golden writer)

- Writes `/tmp/par_golden.bin` (magic `MMPAR001`, ~3.8 MB): 1 prefill + 16
  decode rounds of the tiny hybrid shape, every op's observables plus the
  per-round KV pool / lin f32 state / conv bf16 tail.
- Asserts CPU-reference coverage of all 18 executable opcodes.
- The golden is the CPU oracle's record; the GPU parity test (§4.6) is its
  consumer.

### par_selfcheck — PASS (parity test host self-check)

- Replays the golden against the CPU reference through the exact code the
  GPU parity test uses (reader, op/pool observable sizing, control-buffer
  fills, token chaining, comparator).
- Result (2026-09-25): `1564 ops, 2108 observables, 2009424 elements —
  all bit-exact`, `PARITY PASSED`. Confirms the golden file round-trips
  through the parity test's own logic with zero deviation.

### Sanitizer results

| Check | engine_smoke | plan_test |
|-------|--------------|-----------|
| ASan (heap overflow / invalid access) | PASS | PASS |
| UBSan | PASS | PASS |
| LeakSan | PASS | see caveat below |

- **engine_smoke is fully clean under ASan+UBSan+LeakSan (exit 0).** The full
  engine lifecycle (create → load → step → destroy) leaks nothing.
- **Caveat — plan_test harness leak (test-only, not an engine bug):**
  `plan_test.c` calls `mm_arena_init_host(&e.a_arena, 1 << 20, "a")` at
  `plan_test.c:99` and never tears the arena down before exit. LeakSanitizer
  therefore reports `1,081,344 byte(s) leaked in 2 allocation(s)` (the 1 MiB
  arena + its 32 KiB header, both from `arena_init_common` in
  `src/alloc/alloc.c`) and aborts with exit 1. All functional checks still pass
  (0 FAIL lines; clean exit when run with `ASAN_OPTIONS=detect_leaks=0`).
  Fix (deferred, one-line): release the arena at the end of `main()` before
  return. The *engine itself* is leak-free per engine_smoke.

---

## 6. Bugs Fixed (this session)

### 6.1 `src/model/tensor_registry.c` — `put_spec()` — FIXED

- **Issue:** used `va_start`/`vsnprintf` despite not being a variadic function.
- **Impact:** undefined behavior; `-Werror` build failure.
- **Resolution:** replaced the variadic formatting path with a direct
  `snprintf(name, size, fmt, (unsigned)li);` (verified in source; `fmt` carries
  at most one `%u` layer index).

### 6.2 `src/engine/engine.c` — `mm_engine_create()` — FIXED

- **Issue:** model shape never initialized — `host_tiny_shape()` existed but was
  never called.
- **Impact:** engine created an invalid runtime model layout.
- **Resolution:** `host_tiny_shape(&e->mc)` is now called (engine.c:257) before
  `mm_model_cfg_finalize(&e->mc)` (engine.c:267).

### 6.3 `src/kv/kv.c` — `mm_kvpool_alloc_block()` / `mm_kvpool_free_block()` — FIXED

- **Issue:** block IDs are allocated in range `1..n_blocks` (inclusive), but
  `block_used[]` / `block_ref[]` were sized `n_blocks`.
- **Impact:** heap-buffer-overflow — a real memory-safety bug that would
  eventually affect GPU builds.
- **Resolution:** arrays allocated as `calloc(n_blocks + 1, ...)`
  (kv.c:76-77) and the free bound check adjusted from `blk < n_blocks` to
  `blk <= n_blocks` (`mm_kvpool_free_block` now requires
  `blk > 0 && blk <= p->n_blocks`).

### 6.4 `include/mimfer/kernels.h` / `src/kernels/kx.h` — C/C++ linkage — FIXED

- **Issue:** the GPU launchers (`kx_op_*`, `kx_cuda_oppref`) are DEFINED in
  the C++ translation unit `src/kernels/cuda/cx.cu` (nvcc), while the
  dispatcher (`src/kernels/kx.c`), the engine and the C test binaries call
  them with C linkage. Without `extern "C"` guards the .cu symbols would be
  mangled and the GPU builds would fail at link time.
- **Resolution:** `kernels.h` and `kx.h` now wrap their APIs in
  `#ifdef __cplusplus extern "C" { ... }` guards, and `cx.cu` includes
  `kx.h` so `kx_cuda_oppref` inherits the C linkage from its prototype.
  Host builds are unaffected (the guard is inert in C). Re-verified: all
  host tests still pass under `-Werror`.

---
## 7. Next Priority

**Completed 2026-09-26 (this session):** engine CLI feature surface —
RoPE tables module (`src/rope/`), CLI flag grammar (`src/flags/`),
weights profiles + cross-field validation (`src/config/`), spec/vision
feature gates in `mm_engine_load` — with host tests `rope_test`,
`flags_test`, `engine_features_test` wired into the `make test` suite
(unit tests → engine smoke → feature gates → parity self-check), and all
documentation synced (README feature-surface section, `docs/dflash2.md`
created, `rope.c` added to every documented link set). Full host suite
green from `make clean`.

**NEXT TASK: external validation on a GPU machine** (this host now has the
real CUDA 13.1 toolkit and the GPU build compiles + links driverless via
toolkit stubs — what is missing is a GPU with the driver to *execute*; the
process for the human validator is fully documented; nothing speculative
remains).

Entry points:
- **`GPU_VALIDATION.md`** — the complete runbook (machine requirements,
  CUDA 12.8+/13.x (build-verified on 13.1.115), the ordered build/run
  procedure with expected outputs,
  parity + smoke workflows, troubleshooting, bug-report template,
  hard invariants).
- **`VALIDATION_CHECKLIST.md`** — the go/no-go gate: build succeeds →
  parity_test builds+passes → gpu_smoke builds+passes → graph mode →
  direct mode → determinism → no CUDA runtime errors → no sanitizer
  findings.
- **`RELEASE_READINESS.md`** — completed/pending subsystems, known risks,
  unverified assumptions, TODO/placeholder audit findings.

Order (full commands in GPU_VALIDATION.md §5–§8):

1. Golden + oracle pre-gate on the GPU machine (host gcc):
   `par_golden` → `PAR GOLDEN WRITTEN`; `par_selfcheck` → `PARITY PASSED`
   (if the self-check fails there, rebuild the golden on that machine —
   the golden is CPU-written and must match the validating machine).
2. `parity_test` (nvcc, `-fmad=false -DMM_WITH_CUDA -arch=sm_120`) →
   expect `PARITY PASSED` (1 bf16 ulp gate on computed ops).
3. `gpu_smoke` (nvcc) → expect `GPU SMOKE PASSED` (graph == direct,
   oracle heads, deterministic, clean device syncs).
4. Optional deeper sweep: `compute-sanitizer` around `/tmp/gpu_smoke`.
5. Only after 2–4 pass on a real GPU: Blackwell optimization work (FMA /
   tensor-core paths) — each must re-pass the parity test.

Constraints (binding):

- Do NOT rewrite the planner.
- Do NOT redesign the engine.
- Do NOT introduce a generic backend abstraction.
- The CUDA path must attach to the existing validated host architecture.
- The CPU reference remains the correctness oracle; the tolerance classes
  (bit-exact / 1 bf16 ulp / 4 f32 ulp) are the gate — see the
  `tests/host/par_golden.c` header.

**Lowest-risk path:** preserve the currently validated architecture and attach
CUDA incrementally.

---

## 8. Session Handoff

| Item | Value |
|------|-------|
| Project health | GREEN |
| Host correctness | VERIFIED |
| Memory safety | VERIFIED |
| Determinism | VERIFIED |
| Planner | COMPLETE |
| CPU path | COMPLETE |
| CUDA runtime + kernels | COMPILED (real nvcc 13.1.115, driverless, 2026-09-26) — runtime layer, device mem, cx.cu launch set; sm_120 fatbin inspected via cuobjdump (all 14 kernels present, zero warnings); graph-capture-legal, round-boundary I/O wired; silicon execution PENDING |
| Parity tooling | COMPLETE + host-VERIFIED (par_golden golden writer; parity_test dual-buildable; self-check passes bit-exact) |
| GPU smoke test | WRITTEN (gpu_smoke.c: graph vs direct, oracle heads, reproducible tokens) |
| Validation docs | COMPLETE — GPU_VALIDATION.md (runbook: requirements, procedure, troubleshooting, report template; all link sets include `src/rope/rope.c`), VALIDATION_CHECKLIST.md (go/no-go; Gate 0 requires `make test` green), RELEASE_READINESS.md (status/risks/audit incl. the new feature rows), docs/dflash2.md (declared speculative-decoding scope); 2026-09-26 |
| README | COMPLETE — README.md: Current Model Support (two NInfer Qwen3.8-27B reference models only), Current Artifact Support (NInfer Artifact V2/V3), Hardware Scope (RTX PRO 4000 Blackwell only; soft-gate is a testing affordance), **Engine & CLI Feature Surface** (flag-by-flag status table: wired / validated scaffolding / validated hook), Extensibility (suckless-inspired, architecture influence not code dependency); 2026-09-26 |
| Feature surface | COMPLETE (host) — RoPE tables (`src/rope/rope.c`, plain + YaRN; plain bit-identical to legacy formula so the golden is unchanged), CLI flag grammar (`src/flags/flags.c`), weights profiles + `mm_engine_cfg_validate` cross-field rules (`src/config/`), `--spec`/`--vision` gates in `mm_engine_load`; tests: `rope_test`, `flags_test`, `engine_features_test` (e2e YaRN q-buffer difference vs plain, plain byte-identical, spec/vision refusals); declared speculative scope: docs/dflash2.md; 2026-09-26 |
| Governance | COMPLETE — .clinerules (mandatory rules: memory, branch, architecture, scope, CUDA parity-first, code quality, documentation, philosophy), .gitignore (build/CUDA/test/editor artifacts ignored; all docs + .clinerules explicitly kept versioned; root Makefile + tests/ un-ignored), git repo with 3 commits on `main`; work branch `feature/cmake-flag-surface` pushed to origin (CMake now exists in the tree — `CMakeLists.txt`, host parity + optional CUDA — so the branch name is no longer a historical artifact; see the Build system row); 2026-09-26 |
| Build system | COMPLETE — root `Makefile` (GNU Make only): `make`/`make host` (7 host bins in `build/`), `make test` (6-stage sequential host suite, fail-fast: plan → rope → flags → engine-smoke → engine-features → parity-selfcheck), `make cuda`/`gpu-parity`/`gpu-smoke` (build-only + printed run commands), `check-nvcc`, `help`, `clean`, `distclean`; `CUDA_ARCH ?= sm_120`; `tests/host/par_selfcheck.c` wrapper to the shared comparator; two GNU Make 4.3 traps fixed (order-only dir prerequisite skip; default goal = first file target — see §4.7); verified end-to-end 2026-09-26; **B1 fix applied 2026-09-26**: GPU recipes now link `src/kernels/cuda/cx.cu` (via `GPU_CU_SRCS`, no `-x c`) — verified by `make -n` expansion; **CMake parity 2026-09-26**: `CMakeLists.txt` — same 7 host binaries via CMake/Ninja, Makefile-identical host flags (`-std=c11 -Wall -Wextra -Werror`, no host `-O*`/`-DNDEBUG`/`gnu11`), `test`/`golden`/`smoke`/`parity` CMake targets green; optional CUDA: `-DMIMFER_ENABLE_CUDA=ON` (+ `-DMIMFER_NVCC_EXTRA` for driverless stub linking), CMake default stays host-only, `--target cuda` builds the GPU binaries (mirrors `make` default / `make cuda`); **nvcc `-x` fix 2026-09-26**: all `-x` flags removed from Makefile + CMake + documented recipes — CUDA 13.1's `-x` is a *global* last-value-wins option (per-source `-x c` had been compiling `cx.cu` as plain C), language is now by file extension; `extern "C"` guards added to 17 public headers for the C++ TU; `make cuda NVCC_EXTRA=-L<toolkit>/lib64/stubs` and the CMake CUDA build verified against the **real nvcc 13.1.115** (zero warnings; fatbin: 14 kernels, sm_120); host build stays hermetic (no CUDA toolkit needed for the default target, suite green) |
| Final build audit | COMPLETE (2026-09-26) — `BUILD_AUDIT.md` (B1 critical GPU-link fix + B10 critical: GPU `parity_test` had no `kx_cpu_oppref` provider; fixed with a `#ifdef MM_WITH_CUDA` reference copy in `tests/cuda/parity_test.c` — host recipe byte-identical, `make test` re-green; B2–B9 documented: 3 unregistered TUs pass `-fsyntax-only` under release flags, no CMake exists, `HDRS` caveat, tmux junk, stale README sentence fixed), `SOURCE_TREE.md` (all 57 tracked files: role/size/registration), `MODULE_DEPENDENCIES.md` (per-TU include graph, module map, cross-TU symbol deps, `MM_WITH_CUDA` split); no missing headers; no dead sources; Make = sole build system and matches the documented verbatim commands |
| GPU execution | PENDING — this host has the real CUDA 13.1 toolkit (nvcc 13.1.115) and the GPU binaries compile + link driverless (toolkit stubs); what is missing is a GPU with the driver. On a GPU machine: `make golden && make parity-selfcheck && make cuda`, then run the printed commands (GPU_VALIDATION.md §4–§8) |
| Recommended next step | Human validation on a GPU machine: GPU_VALIDATION.md §4–§8, gated by VALIDATION_CHECKLIST.md (now drivable via `make` targets, §4.7) |
| Audit findings | Final build audit 2026-09-26 (`BUILD_AUDIT.md` B1–B10): B1+B10 GPU-link defects FIXED (see above); artifact/tokenizer/telemetry written but unintegrated — all three pass `-fsyntax-only` under the exact release flags (not latent breakage); **no missing headers, no dead sources** (every TU registered, or the 3 above); zero TODO/FIXME markers; ~1.3 GB tmux logs at repo root (junk, R6, human-decision); docs/architecture.md + docs/design.md referenced but absent; README stale Extensibility sentence fixed (B9); README docs table + RELEASE_READINESS §7 synced to the three new audit docs |
| Risk | Compile/link layer is CLOSED: `cx.cu` + the full GPU link set compile and link clean against the real nvcc 13.1.115 (driverless, 2026-09-26, both build systems, zero warnings, fatbin verified). Remaining risk is first **silicon** execution: graph capture, launch attributes, L2 windows, driver-reported device attributes (B1/B10 GPU-link defects already fixed). Any runtime failure is isolated by the runbook's build order and reportable per the template |
