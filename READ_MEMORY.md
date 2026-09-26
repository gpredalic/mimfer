# READ_MEMORY.md — mimfer

> **Purpose:** Single source of truth for project state. Read this first before
> touching any code. Update it immediately whenever project state materially
> changes.
>
> **Last updated:** 2026-09-26 (session handoff: external validation pack + README.md
> complete — GPU_VALIDATION.md + VALIDATION_CHECKLIST.md + RELEASE_READINESS.md
> + README.md; GPU execution pending)
> **Project health:** GREEN
> **Host correctness:** VERIFIED · **Memory safety:** VERIFIED · **Determinism:** VERIFIED
> **GPU parity:** TOOLING COMPLETE, host self-check VERIFIED · **GPU execution:** PENDING (no nvcc/GPU on this host)
> **External validation pack:** COMPLETE (GPU_VALIDATION.md runbook, VALIDATION_CHECKLIST.md go/no-go, RELEASE_READINESS.md status/risks)

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
| [COMPLETE] | End-to-end GPU smoke test (written; pending GPU execution) |
| [COMPLETE] | External validation pack: GPU_VALIDATION.md (runbook), VALIDATION_CHECKLIST.md (go/no-go), RELEASE_READINESS.md (status/risks/audit) |
| [COMPLETE] | README.md (project README: current model support, artifact support, hardware scope, extensibility) |
| [COMPLETE] | Project governance: .clinerules (mandatory rules), .gitignore, git repo initialized (main) |
| [PENDING]  | GPU execution of parity + smoke tests (needs nvcc + a GPU) |
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
                  telemetry.h, tokenizer.h, mimfer.h)
src/engine/       engine.c          — runtime orchestration (create/load/step)
src/plan/         plan.c            — planner (isolated; owns op-graph + activation planning)
src/kernels/      kx.c, kx.h        — kernel dispatcher (mm_kx_invoke)
src/kernels/cpu/  cx.c              — CPU reference kernels
src/model/        tensor_registry.c/.h — tensor registry (NOT part of planner)
src/kv/           kv.c              — KV block pool (block IDs are 1..n_blocks)
src/sched/        sched.c           — request scheduler (submit/active/queued)
src/sampling/     sampling.c        — sampling
src/alloc/        alloc.c           — arena allocator (mm_arena_init_host, ...)
src/config/       config.c          — model cfg finalize
src/core/         mimfer.c          — core utilities
src/cuda/         cuda_rt.c         — CUDA runtime layer (MM_WITH_CUDA guarded; host no-op stubs)
                  cuda_mem.c        — device arenas + pinned host I/O block (MM_WITH_CUDA)
src/kernels/cuda/ cx.cu             — GPU kernel launch set (nvcc, -DMM_WITH_CUDA, -fmad=false)
tests/host/       plan_test.c, engine_smoke.c, par_golden.c (golden writer)
tests/cuda/       parity_test.c (GPU comparator; host self-check build),
                  gpu_smoke.c (end-to-end GPU smoke test)
```

---
## 4. Verified Build Commands (KNOWN-GOOD)

Both commands re-verified **2026-09-25** from the project root
(`/home/razvijalec/mimfer`): clean compile under `-Werror`, tests pass.

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
    src/kernels/kx.c src/model/tensor_registry.c src/sched/sched.c \
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
    src/kernels/kx.c src/model/tensor_registry.c src/sched/sched.c \
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
    src/model/tensor_registry.c src/sched/sched.c -lm \
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
    src/model/tensor_registry.c src/sched/sched.c -lm \
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

### 4.6 CUDA builds — PENDING GPU VERIFICATION (written + syntax-checked only)

This host has **no nvcc / CUDA toolkit / GPU**; the commands below are the
documented build for a GPU machine and are marked UNVERIFIED until run there.
They were syntax-checked on this host with `gcc -fsyntax-only` (C sources,
`-DMM_WITH_CUDA`, stub `cuda_runtime.h` exposing `cudaStream_t`); `cx.cu`
itself has never been compiled.

**Canonical runbook: `GPU_VALIDATION.md`** (machine/CUDA/compiler
requirements, the full ordered procedure incl. the `par_selfcheck` oracle
pre-gate, expected outputs, troubleshooting, bug-report template). The
commands in that document add `-arch=sm_120` (PRO 4000 Blackwell; replace
per card) and document `MIMFER_SOFT_DEVICE_GATE=1` for non-target cards —
the commands below predate those two additions and still work on the
target card with a 12.8+ toolkit whose default arch covers sm_120.
Go/no-go gate: `VALIDATION_CHECKLIST.md`.

```sh
# Parity test (numeric gate): compares GPU op outputs to the CPU golden.
# Run on the GPU machine after building /tmp/par_golden.bin on ANY host
# (the golden is CPU-written and machine-independent):
nvcc -O2 -fmad=false -DMM_WITH_CUDA \
     -Iinclude -Isrc/model -Isrc/kernels \
     -x c tests/cuda/parity_test.c \
     -x c src/engine/engine.c -x c src/plan/plan.c \
     -x c src/alloc/alloc.c -x c src/config/config.c \
     -x c src/core/mimfer.c -x c src/cuda/cuda_rt.c \
     -x c src/cuda/cuda_mem.c -x c src/sampling/sampling.c \
     -x c src/kv/kv.c -x c src/kernels/kx.c \
     src/kernels/cuda/cx.cu \
     -x c src/model/tensor_registry.c -x c src/sched/sched.c \
     -o /tmp/parity_test
/tmp/parity_test /tmp/par_golden.bin

# End-to-end GPU smoke test (graph + direct dispatch, oracle heads):
nvcc -O2 -fmad=false -DMM_WITH_CUDA \
     -Iinclude -Isrc/model -Isrc/kernels \
     -x c tests/cuda/gpu_smoke.c \
     -x c src/engine/engine.c -x c src/plan/plan.c \
     -x c src/alloc/alloc.c -x c src/config/config.c \
     -x c src/core/mimfer.c -x c src/cuda/cuda_rt.c \
     -x c src/cuda/cuda_mem.c -x c src/sampling/sampling.c \
     -x c src/kv/kv.c -x c src/kernels/kx.c \
     src/kernels/cuda/cx.cu \
     -x c src/model/tensor_registry.c -x c src/sched/sched.c \
     -o /tmp/gpu_smoke
/tmp/gpu_smoke
```

- **`-fmad=false` is mandatory** for `cx.cu`: the CPU reference (baseline
  x86-64) emits mul+add only; allowing FMA contraction on the GPU would
  change f32 accumulation rounding and blow the 1-ulp bf16 parity gate.
- Linkage note: `kernels.h` / `kx.h` pin C linkage on the dispatcher ABI
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

---
## 5. Verified Tests (re-verified 2026-09-25)

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

**NEXT TASK: external validation on a GPU machine** (this host has no nvcc /
CUDA toolkit / GPU — the code is written, the tooling is verified on the
host, the execution is what is missing). The process for the human
validator is fully documented; nothing speculative remains.

Entry points:
- **`GPU_VALIDATION.md`** — the complete runbook (machine requirements,
  CUDA 12.8+, the ordered build/run procedure with expected outputs,
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
| CUDA runtime + kernels | WRITTEN (runtime layer, device mem, cx.cu launch set; graph-capture-legal, round-boundary I/O wired) |
| Parity tooling | COMPLETE + host-VERIFIED (par_golden golden writer; parity_test dual-buildable; self-check passes bit-exact) |
| GPU smoke test | WRITTEN (gpu_smoke.c: graph vs direct, oracle heads, reproducible tokens) |
| Validation docs | COMPLETE — GPU_VALIDATION.md (runbook: requirements, procedure, troubleshooting, report template), VALIDATION_CHECKLIST.md (go/no-go), RELEASE_READINESS.md (status/risks/audit); 2026-09-26 |
| README | COMPLETE — README.md created: Current Model Support (two NInfer Qwen3.8-27B reference models only), Current Artifact Support (NInfer Artifact V2/V3), Hardware Scope (RTX PRO 4000 Blackwell only; soft-gate is a testing affordance), Extensibility (suckless-inspired, architecture influence not code dependency); 2026-09-26 |
| Governance | COMPLETE — .clinerules (mandatory rules: memory, branch, architecture, scope, CUDA parity-first, code quality, documentation, philosophy), .gitignore (build/CUDA/test/editor artifacts ignored; all docs + .clinerules explicitly kept versioned), git repo initialized on main (no commits yet); 2026-09-26 |
| GPU execution | PENDING — no nvcc/GPU on this host; run the GPU_VALIDATION.md procedure on a GPU machine |
| Recommended next step | Human validation on a GPU machine: GPU_VALIDATION.md §4–§8, gated by VALIDATION_CHECKLIST.md |
| Audit findings | RELEASE_READINESS.md §7: zero TODO/FIXME markers; artifact/tokenizer/telemetry written but unintegrated; 6.4 GB tmux logs at repo root (junk); docs/architecture.md + docs/design.md referenced but absent |
| Risk | cx.cu has never been compiled (no nvcc here); first GPU run may surface .cu-specific compile issues — everything around it is verified; failures are isolated by the runbook's build order and are reportable per the template |
