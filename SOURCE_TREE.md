# SOURCE_TREE.md — mimfer annotated source inventory

> **Date:** 2026-09-26 (final build audit; see `BUILD_AUDIT.md`).
> Every tracked file is listed with its role, size, and build
> registration. "Build registration" = the Makefile source set(s) the
> file is compiled in:
> `PLAN` = `PLAN_SRCS` · `ENG` = `ENGINE_SRCS` · `CFG` = `CFG_TEST_SRCS`
> · `GPU` = `GPU_SRCS` (host set minus `cpu/cx.c`, plus `cuda/cx.cu`) ·
> `TEST` = a test-binary-only set. A file may appear in several sets.
> **Total: 57 tracked files** (17 public headers, 2 in-tree private
> headers, 19 library TUs, 9 test TUs, 9 root files, 1 docs file).

## 1. Root files

| File | Role |
|------|------|
| `Makefile` | Canonical build system (GNU Make only; no CMake — BUILD_AUDIT.md B4). Host + GPU recipes, `make test` 6-stage suite, `check-nvcc` gate, `CUDA_ARCH ?= sm_120`. |
| `.gitignore` | Keeps `build/`, CUDA artifacts, editor/OS junk, `*.log` out of git; deliberately keeps all docs + sources versioned. Carries template CMake/Ninja patterns (unused; B4). |
| `.clinerules` | Mandatory project governance (memory, branch, architecture, scope, CUDA parity-first, code quality, documentation, philosophy). |
| `LICENSE` | License. |
| `README.md` | Project README: model/artifact/hardware support, engine & CLI feature surface, extensibility, documentation index, build quick-start. |
| `READ_MEMORY.md` | Project state source of truth; read first / update last (governance rule). |
| `GPU_VALIDATION.md` | Hardware validation runbook: requirements, build/run procedure (verbatim `gcc`/`nvcc` commands), troubleshooting, report template. |
| `VALIDATION_CHECKLIST.md` | Go/no-go checklist for the first GPU run (Gate 0: `make test` green). |
| `RELEASE_READINESS.md` | Completed vs pending subsystems, known risks (R1–R7), unverified assumptions (A1–A7), audit results. |
| `BUILD_AUDIT.md` | This audit's findings (B1–B9), conformance matrix, evidence. |
| `SOURCE_TREE.md` | This file. |
| `MODULE_DEPENDENCIES.md` | Per-TU include graph and module dependency map. |

## 2. Public API headers — `include/mimfer/` (17)

Included as `#include "mimfer/<name>.h"`; `-Iinclude`.

| Header | Lines | Owns | Notable dependents |
|--------|------:|------|--------------------|
| `mimfer.h` | 153 | Root: common types, status enum, logging, version | all other public headers |
| `config.h` | 196 | `mm_engine_cfg`, validation, weights profiles, policy | plan, engine, kv, rope, flags + tests |
| `plan.h` | 106 | Planner API, op/plan types | kernels.h, engine.h, kx.h, plan.c |
| `engine.h` | 157 | Engine API; the hub — includes 12 other public headers | engine.c, kx.c/kx.h, cpu/cx.c, cuda/cx.cu, plan.c, tensor_registry.c/.h, all engine-level tests |
| `kernels.h` | 132 | Dispatcher contract: `mm_kx_invoke`, `kx_op_*` launchers, `mm_kcall`, `mm_ctrl` | plan.h, cuda_mem.h, engine.h, kx.c, cpu/cx.c, cuda/cx.cu, rope.h, plan_test |
| `kv.h` | 132 | KV block allocation | engine.h, kv.c |
| `tensor.h` | 156 | Tensor registry API / descriptor | engine.h, mimfer.c, cpu/cx.c, cuda/cx.cu, tensor_registry.c/.h |
| `alloc.h` | 95 | Arena allocator | engine.h, kv.h, alloc.c |
| `cuda_rt.h` | 104 | CUDA runtime layer — **public surface is CUDA-free** (`<cuda_runtime.h>` only under `#ifdef MM_WITH_CUDA`, B5) | plan.h, kv.c, config.c, cuda_rt.c, cuda_mem.h, tests/cuda |
| `cuda_mem.h` | 37 | Device arenas + pinned I/O block | cuda_mem.c, engine.c |
| `sampling.h` | 42 | Sampler policy | config.c, sampling.c, engine.h |
| `sched.h` | 73 | Scheduler | engine.h, sched.c |
| `rope.h` | 69 | RoPE frequency tables (plain + YaRN) | rope.c, engine.h, rope_test |
| `flags.h` | 37 | CLI flag grammar (single table + validation) | flags.c, flags_test only (B3) |
| `artifact.h` | 142 | NInfer artifact V2/V3 loader API | engine.h, artifact.c (unregistered, B2) |
| `tokenizer.h` | 63 | Tokenizer interface | engine.h, tokenizer.c (unregistered, B2) |
| `telemetry.h` | 49 | Telemetry counters | engine.h, telemetry.c (unregistered, B2) |

**Orphan check:** every public header is included by at least one TU
(verified from the full include graph); `mimfer.h` is the root of the
header DAG — no cycles among public headers (`engine.h` sits at the top
of the public include order but nothing it includes points back).

## 3. Library sources — `src/` (19 TUs + 2 private headers)

| File | Lines | Module | Build registration | Notes |
|------|------:|--------|--------------------|-------|
| `src/core/mimfer.c` | 85 | core (status/logging/version) | PLAN, ENG, CFG | |
| `src/config/config.c` | 298 | config/validation/profiles | PLAN, ENG, CFG | `<cuda_runtime.h>` guarded by `MM_WITH_CUDA` |
| `src/alloc/alloc.c` | 239 | arena allocator | PLAN, ENG | same guard |
| `src/sampling/sampling.c` | 39 | sampling policy | PLAN, ENG, CFG | |
| `src/cuda/cuda_rt.c` | 417 | CUDA runtime layer | PLAN, ENG, GPU | host branch = honest no-op stubs; real CUDA under `MM_WITH_CUDA` |
| `src/cuda/cuda_mem.c` | 43 | device arenas + pinned I/O | ENG, GPU | same split |
| `src/plan/plan.c` | 506 | planner | PLAN, ENG, GPU | |
| `src/kv/kv.c` | 390 | KV block allocation | ENG, GPU | |
| `src/sched/sched.c` | 96 | scheduler | ENG, GPU | |
| `src/rope/rope.c` | 134 | RoPE tables (plain + YaRN) | ENG, GPU + `ROPE_TEST_SRCS` | plain tables bit-identical to legacy formula |
| `src/flags/flags.c` | 311 | CLI flag grammar | `FLAGS_TEST_SRCS` only (B3) | consumed by a future CLI; full unit test |
| `src/engine/engine.c` | 670 | engine (load/decode/feature gates) | ENG, GPU | largest TU; includes the 12-header hub `engine.h` + private `tensor_registry.h` + `cuda_mem.h` |
| `src/kernels/kx.c` | 47 | dispatcher (`mm_kx_invoke` switch) | ENG, GPU | calls `kx_op_*` (defined by `cpu/cx.c` or `cuda/cx.cu` per build) |
| `src/kernels/cpu/cx.c` | 490 | CPU reference kernels (18 `kx_op_*`) | ENG (host only) | golden reference; excluded from GPU set |
| `src/kernels/cuda/cx.cu` | 1042 | CUDA kernel launch set (18 `kx_op_*` + `kx_cuda_oppref`) | GPU only (nvcc, by extension — **no `-x`**) | the only C++/CUDA TU; **now compiles + links clean on nvcc 13.1.115 (driverless, 2026-09-26)**; **B1 fix** ensures it is on the nvcc link line |
| `src/model/tensor_registry.c` | 238 | tensor registry impl | ENG, GPU | |
| `src/artifact/artifact.c` | 448 | artifact V2/V3 loader | **none** (B2) | `-fsyntax-only` clean under release flags |
| `src/tokenizer/tokenizer.c` | 285 | tokenizer | **none** (B2) | same |
| `src/telemetry/telemetry.c` | 118 | telemetry | **none** (B2) | same |

In-tree private headers (not under `include/`; reachable via
`-Isrc/kernels`, `-Isrc/model`; both listed in `HDRS`):

| File | Lines | Role |
|------|------:|------|
| `src/kernels/kx.h` | 49 | CPU/CUDA dispatch internals: `kx_cpu_oppref` / `kx_cuda_oppref` prototypes; C-linkage guard shared by the C reference and the `.cu` |
| `src/model/tensor_registry.h` | 37 | tensor registry internal API (used by `engine.c` and `tensor_registry.c`) |

## 4. Tests — `tests/` (9 TUs)

| File | Lines | Built by | Covers |
|------|------:|----------|--------|
| `tests/host/plan_test.c` | 209 | `build/plan_test` (`PLAN_SRCS`) | planner correctness; part of `make test` |
| `tests/host/rope_test.c` | 209 | `build/rope_test` (`ROPE_TEST_SRCS`) | RoPE tables vs the reference formula (plain + YaRN); part of `make test` |
| `tests/host/flags_test.c` | 295 | `build/flags_test` (`FLAGS_TEST_SRCS`) | full CLI flag grammar + cross-field validation; part of `make test` |
| `tests/host/engine_smoke.c` | 152 | `build/engine_smoke` (`ENGINE_SRCS`) | engine end-to-end on CPU; also the token-head oracle for `gpu_smoke`; part of `make test` |
| `tests/host/engine_features_test.c` | 232 | `build/engine_features` (`ENGINE_SRCS`) | e2e YaRN q-buffer difference vs plain (plain byte-identical), `--spec`/`--vision` refusals; part of `make test` |
| `tests/host/slot_lifecycle_test.c` | 256 | `build/slot_lifecycle_test` (`ENGINE_SRCS`) + `build/gpu_slot` (GPU set, nvcc — the same source) | multi-request slot teardown regression: A finishes → slot torn down (KV released, linear/conv state reset, block table zeroed) → B runs on the same slot and matches a fresh engine's B stream; part of `make test`; GPU run: `make gpu-slot` (GPU_VALIDATION.md §8) |
| `tests/host/par_golden.c` | 449 | `build/par_golden` (`ENGINE_SRCS`) | golden writer (1 prefill + 16 decode, seed 12345) |
| `tests/host/par_selfcheck.c` | 19 | `build/par_selfcheck` (`ENGINE_SRCS`) | thin wrapper that `#include`s `../cuda/parity_test.c` (one comparator source of truth); part of `make test` |
| `tests/cuda/parity_test.c` | 805 | `build/parity_test` (GPU set, nvcc, by extension; also `#include`d by `par_selfcheck.c`) | per-op GPU↔golden parity, 18 opcodes, 3 tolerance classes; dual-buildable (host self-check + GPU); carries the `MM_WITH_CUDA`-guarded `kx_cpu_oppref` reference copy for the GPU link (B10) |
| `tests/cuda/gpu_smoke.c` | 227 | `build/gpu_smoke` (GPU set, nvcc, by extension) | end-to-end GPU: graph vs direct, determinism, CPU-oracle token heads; `#error` without `MM_WITH_CUDA` |

## 5. Untracked / ignored

- `build/` — all build artifacts (git-ignored; removed by `make clean`).
- `tmux-client-153455.log` (3.5 KB), `tmux-out-153457.log` (11 MB),
  `tmux-server-153457.log` (1.3 GB) — session junk at the repo root,
  git-ignored by `*.log`, deletion is a human decision (B7 / R6).

## 6. Registration summary (answers to the audit questions)

- **Missing headers:** none — every `#include` in every TU resolves
  against `-Iinclude -Isrc/model -Isrc/kernels` plus the local directory
  (full graph in `MODULE_DEPENDENCIES.md`).
- **Missing source registrations (in a build):** `artifact.c`,
  `tokenizer.c`, `telemetry.c` — intentional, written-but-unintegrated
  (B2); all three pass `-fsyntax-only` under the exact release flags.
- **Dead source files (referenced by nothing):** none — every TU is in
  at least one build, or (the three above) has its header included by
  `engine.h`. `flags.c` is test-registered only by design (B3).
- **Modules not included in builds:** the same three modules
  (artifact/tokenizer/telemetry) — the only ones.
- **CMake vs Make:** no CMake exists; the Makefile is the sole build
  system and the documented verbatim commands match it (B4).

