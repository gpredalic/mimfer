# READ_MEMORY.md — mimfer

> **Purpose:** Single source of truth for project state. Read this first before
> touching any code. Update it immediately whenever project state materially
> changes.
>
> **Last updated:** 2026-09-27 (slot-lifecycle teardown fix S1/S2/S3
> VERIFIED on CPU + the RTX PRO 4000 Blackwell — `SLOT LIFECYCLE PASSED`,
> reused-slot B byte-identical to a fresh engine; `bugfix/slot-lifecycle`
> holds the red-test + fix + runbook commits; merging the three bugfix
> branches is next; earlier same day: GPU parity + smoke VERIFIED —
> `PARITY PASSED` + `GPU SMOKE PASSED`; the device-gate VRAM bug is FIXED
> and re-verified on silicon WITHOUT the bypass — hard gate
> `mm_device_check ok (target profile matched)`, committed on
> `bugfix/vram-gate-tolerance`):
> (1) GPUs: 2× NVIDIA RTX PRO 4000 Blackwell, PCI 0000:05:00.0 +
> 0000:06:00.0, kernel module 595.71.05; container userspace driver libs
> 595.91.07 (apt nvidia-utils-595) — nvidia-smi reports an NVML
> version mismatch, but this does NOT affect the CUDA driver API:
> cuInit / enumeration / context / real H2D+D2H round-trip all VERIFIED
> with a minimal driver-API probe (sm_120, 70 SMs).
> (2) GPU split: CUDA dev0 = PCI 05:00.0 (/dev/nvidia1) is BUSY —
> cuCtxCreate/primaryCtxRetain → OOM (VRAM held by a workload outside
> this container, presumably ai-dev); CUDA dev1 = PCI 06:00.0
> (/dev/nvidia0) is FREE (total 23.43 GiB, ~23.2 GiB free). ALWAYS run
> with `CUDA_VISIBLE_DEVICES=1` (or UUID GPU-a9bf78f7-3ecf-9f3a-9caa-8a9be63ef1be).
> (3) Local toolkit: /home/razvijalec/tools/cuda/usr/local/cuda-13.1 —
> real nvcc V13.1.115 + ptxas/cuobjdump/nvdisasm; NO compute-sanitizer or
> nsight tools; no /usr/local/cuda; /tmp/fake-cuda/bin/nvcc is a CMake
> dry-run stub — never use it.
> `make cuda NVCC=/home/razvijalec/tools/cuda/usr/local/cuda-13.1/bin/nvcc`
> rebuilt both GPU binaries from clean (verified, sm_120 cubin inside
> both, zero warnings). (4) GPU PARITY VERIFIED ON SILICON (runs 5–6,
> 2026-09-27): both run-2 blockers cleared — the NULL `g_dev` dst got
> the one-line `buf_grow` fix (kept), and the graph-mode IMA was NOT a
> graph bug: 1-based KV block ids were used as 0-based pool indices
> (first allocated block id 2 = index 2 = one past the last block with
> n_blocks=2); the device L7 V row then wrote 256 B past the KV pool
> into the block-table slot, corrupting blk_tab_dev → IMA on the next
> attention op. Fixed at all four index sites (cpu kv_row/kv_row_w,
> cuda k_kvstore/kv_row_dev). A second, fix-unblocked gate failure: the
> LIN decay constant was computed in-kernel with device expf (1 f32 ulp
> off glibc) and the 1-ulp drift compounded over the recurrence — decay
> is now computed host-side (same glibc call as the CPU reference) and
> passed to k_lin. Temporary IMA/D2H instrumentation removed.
> RESULTS: `PARITY PASSED` (1564 ops, 2108 observables, 2,009,424
> elements, 2,009,422 bit-exact; class-2 f32 state bit-exact) and
> `GPU SMOKE PASSED` (graph == direct, both oracle heads), run with
> `CUDA_VISIBLE_DEVICES=1 MIMFER_SOFT_DEVICE_GATE=1`. Host suite green;
> `par_golden` regenerated after the KV fix (pool layout moved, values
> and sampled heads unchanged). Committed on `bugfix/kv-index-ima`
> (3 commits). DEVICE-GATE VRAM BUG FIXED 2026-09-27: the driver reports
> 23.4256 GiB (2.39% below nominal) on the 24 GB card (driver/firmware
> reserve — verified with a driver-API probe); the gate now tolerates up
> to 5% below nominal (`MM_VRAM_TOLERANCE_PCT`) + regression test
> `device_gate_test` (host suite green under Make AND CMake); silicon
> re-run WITHOUT the bypass: `mm_device_check ok (target profile
> matched)`, `GPU SMOKE PASSED` + `PARITY PASSED` (identical bit-exact
> results). Committed on `bugfix/vram-gate-tolerance`. SLOT-LIFECYCLE
> TEARDOWN FIXED + VERIFIED 2026-09-27: a finished sequence's slot was
> being recycled WITHOUT teardown — (S1) KV blocks never released
> (`mm_kvpool_release_slot` had no callers; a pool leak per sequence),
> (S2) the linear recurrence/conv state carried the finished sequence
> into the next one (`mm_linstate_reset` had no callers), (S3) the
> block-table row kept stale block ids until the next prefill's
> incremental push overwrote them. `teardown_slot()` now runs at
> sequence completion in `step_decode` (release + reset + zeroed-row
> push, at the round boundary after the token readback host sync).
> Regression `slot_lifecycle_test` (one shared host/GPU source; A =
> 40+40 tokens spanning two KV blocks, then B on the freed slot): RED
> pre-fix (S1/S2/S3 all FAIL), GREEN post-fix on host AND sm_120 silicon
> — `SLOT LIFECYCLE PASSED` (reused-slot B stream byte-identical to the
> fresh-engine B and to the host suite; pool free count back to
> post-load after A and B; A's block-table row zeroed; deterministic
> across two GPU runs). Committed on `bugfix/slot-lifecycle` (3 commits:
> red test, fix, docs). NEXT: merge the three bugfix
> branches to main (human gate), the pinned-ctrl H2D design
> cleanup, then Blackwell optimizations (each must re-pass parity)
> **Project health:** GREEN (host) · GPU execution VERIFIED (parity + smoke + slot-lifecycle PASSED on sm_120)
> **Host correctness:** VERIFIED · **Memory safety:** VERIFIED · **Determinism:** VERIFIED
> **GPU build:** VERIFIED (real nvcc 13.1.115, sm_120 fatbin inspected, zero warnings) · **GPU execution:** VERIFIED 2026-09-27 — `PARITY PASSED` (2,009,422/2,009,424 elements bit-exact; f32 state bit-exact) + `GPU SMOKE PASSED` (graph == direct) + `SLOT LIFECYCLE PASSED` (teardown fix S1/S2/S3); run-2 blockers fixed (NULL `g_dev` dst, KV block-id IMA); device-gate VRAM bug FIXED 2026-09-27 (5% tolerance `MM_VRAM_TOLERANCE_PCT` + `device_gate_test`) and re-verified on silicon WITHOUT `MIMFER_SOFT_DEVICE_GATE` (hard gate `mm_device_check ok (target profile matched)`)
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
| [COMPLETE] | CUDA Kernels (GPU-VERIFIED 2026-09-27: parity PASSED on sm_120, f32 state bit-exact) |
| [COMPLETE] | CPU/CUDA parity tooling (golden writer + parity test + self-check) |
| [COMPLETE] | RoPE frequency tables (plain + YaRN NTK-by-parts) — `src/rope/rope.c` |
| [COMPLETE] | CLI flag surface (one grammar table + cross-field validation) — `src/flags/flags.c` |
| [COMPLETE] | Weights profiles (quasar / neroued context defaults + artifact metadata) |
| [COMPLETE] | Engine feature gates: `--spec` refusal, `--vision` hook, `--weights-profile` |
| [COMPLETE] | Engine feature test harness (`engine_features_test.c`: YaRN e2e + refusals) |
| [COMPLETE] | End-to-end GPU smoke test (GPU-VERIFIED 2026-09-27: graph == direct, oracle heads) |
| [COMPLETE] | External validation pack: GPU_VALIDATION.md (runbook), VALIDATION_CHECKLIST.md (go/no-go), RELEASE_READINESS.md (status/risks/audit) |
| [COMPLETE] | README.md (project README: current model support, artifact support, hardware scope, extensibility) |
| [COMPLETE] | Project governance: .clinerules (mandatory rules), .gitignore, git repo initialized (main) |
| [COMPLETE] | Canonical build system: root Makefile (GNU Make only) — host + CUDA build rules, `make` / `make test` / `make cuda` / `make help` / `make clean` / `make distclean`; artifacts in `build/` |
| [COMPLETE] | Final build audit (2026-09-26): `BUILD_AUDIT.md` (findings B1–B10, conformance matrix, evidence), `SOURCE_TREE.md` (annotated file inventory), `MODULE_DEPENDENCIES.md` (include graph + module map); two critical GPU-link defects fixed (B1, B10) |
| [COMPLETE] | GPU execution of parity + smoke tests — VERIFIED on silicon 2026-09-27 (RTX PRO 4000 Blackwell, sm_120, CUDA 13.1.115, driver 595.71.05): `PARITY PASSED` (1564 ops / 2,009,424 elements, 2,009,422 bit-exact; f32 state bit-exact) + `GPU SMOKE PASSED` (graph+direct). Run-2 blockers fixed on `bugfix/kv-index-ima`: D2H scratch `buf_grow` (§6.7), 1-based KV block-id indexing = the IMA root cause (§6.5), host-supplied LIN decay (§6.6); temporary instrumentation removed. Device-gate VRAM bug fixed 2026-09-27 (5% tolerance + `device_gate_test` regression, §6.8) and re-verified on silicon WITHOUT the bypass (hard gate `mm_device_check ok (target profile matched)`) |
| [COMPLETE] | Device-gate VRAM tolerance — `mm_device_check` accepts the driver-reported total within 5% below nominal (the 24 GB card reports 23.4256 GiB, measured 2026-09-27) + `device_gate_test` regression (host suite, both build systems); silicon re-run without `MIMFER_SOFT_DEVICE_GATE` passed the hard gate natively |
| [COMPLETE] | Slot-lifecycle teardown (S1+S2+S3) — `teardown_slot()` in `src/engine/engine.c` runs at sequence completion: KV blocks released to the pool (S1, no leak), the slot's linear/conv state reset (S2), the block-table row zeroed and pushed (S3, the "no block" state). Regression `slot_lifecycle_test` / `gpu_slot` (one shared source, host + GPU): RED pre-fix, GREEN post-fix on host AND silicon 2026-09-27 (`SLOT LIFECYCLE PASSED`; reused-slot B byte-identical to a fresh engine; pool free count back to post-load; block-table row zeroed; deterministic across two GPU runs) |
| [PENDING]  | Blackwell Optimizations |

Note: the CUDA execution path is written in full — `src/cuda/cuda_rt.c`
(streams, events, graph capture, error-checked API wrappers, honest no-op
host stubs), `src/cuda/cuda_mem.c` (device arenas + pinned I/O block), and
`src/kernels/cuda/cx.cu` (the full GPU kernel launch set, parity-first
mirrors of the CPU reference, built with `-fmad=false`). It is now
**EXECUTION-VERIFIED on real silicon** (2026-09-27, RTX PRO 4000 Blackwell
sm_120): the parity test gates all 18 executable opcodes at the tolerance
classes (2,009,422 of 2,009,424 elements bit-exact; the f32 linear-attention
state bit-exact) and the smoke test matches the host oracle heads in both
graph and direct mode. `tests/cuda/parity_test.c` additionally builds on
the host as a self-check (§4.5) and PASSES there (bit-exact) — the oracle
pre-gate for the GPU run.

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
                  device_gate_test.c (device-gate policy: VRAM tolerance
                  regression-anchored to the measured driver value 23.4256 GiB),
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

### 4.6 CUDA builds — BUILD- + SILICON-VERIFIED (toolkit 2026-09-26; GPU runs 2026-09-27)

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

**Silicon results (2026-09-27, RTX PRO 4000 Blackwell, sm_120, CUDA 13.1.115, driver 595.71.05 — run via the Makefile: `make golden && make gpu-parity && make gpu-smoke`, then `CUDA_VISIBLE_DEVICES=1 MIMFER_SOFT_DEVICE_GATE=1` on the rebuilt binaries):**

- `build/parity_test build/par_golden.bin` → **PARITY PASSED** (1 prefill + 16 decode rounds; 1564 ops, 2108 observables, 2,009,424 elements, 2,009,422 bit-exact; class-1 max 1.000 bf16 ulp on 2 elements; class-2 f32 state max 0 ulp = bit-exact).
- `build/gpu_smoke` → **GPU SMOKE PASSED** (both oracle heads, graph == direct).
- Two gate failures found and fixed during this run (see §6.5–§6.7): the KV block-id IMA (which aborted every previous run at prefill op81, before any decode comparison) and the LIN decay constant (device `expf` 1 f32 ulp off glibc; compounded over the recurrence; now host-supplied).
- `build/par_golden.bin` was REGENERATED after the KV index fix (the fix moves pool bytes, not values — sampled-token heads are unchanged, so the `gpu_smoke` hardcoded oracles stayed valid).
- `compute-sanitizer` is still unavailable in the local toolkit (see the §4.6 note) — the parity gate is the numeric verifier.
- Re-run 2026-09-27 AFTER the device-gate VRAM fix (§6.8), WITHOUT `MIMFER_SOFT_DEVICE_GATE`: hard gate `mm_device_check ok (target profile matched)`, `GPU SMOKE PASSED` (both oracle heads, `rounds=17`/`rounds=71`) + `PARITY PASSED` (2,009,422/2,009,424 bit-exact — identical to the bypassed run). Probe line: `NVIDIA RTX PRO 4000 Blackwell (sm_120, 70 SMs, 23 GiB, 672 GB/s derived, l2 48 MiB, smem/SM 100 KiB)` — derived bandwidth exactly 672 GB/s; L2 reports 48 MiB vs the 96 MiB ESTIMATE (soft check, log-only, never fatal).

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
| `make` (default) | build all host targets (same as `make host`) → `build/plan_test`, `build/engine_smoke`, `build/rope_test`, `build/flags_test`, `build/engine_features`, `build/slot_lifecycle_test`, `build/par_golden`, `build/par_selfcheck` |
| `make host` | same as default |
| `make test` / `make tests` | run the host suite strictly sequentially (fail-fast): `plan-test` → `rope-test` → `flags-test` → `engine-smoke` → `engine-features` → `slot-lifecycle` → `parity-selfcheck` (the last builds `par_golden`, writes `build/par_golden.bin`, then replays it) — unit tests first, engine smoke, feature gates, slot teardown, parity oracle gate last |
| `make plan-test` / `make rope-test` / `make flags-test` / `make engine-smoke` / `make engine-features` / `make slot-lifecycle` / `make golden` / `make parity-selfcheck` | single test targets (aliases: `make parity` → parity-selfcheck, `make smoke` → engine-smoke) |
| `make cuda` | build `build/parity_test` + `build/gpu_smoke` + `build/gpu_slot` with nvcc (needs nvcc on PATH; prints a run command, never runs GPU tests; driverless hosts: `make cuda NVCC_EXTRA=-L<toolkit>/lib64/stubs`) |
| `make gpu-parity` / `make gpu-smoke` / `make gpu-slot` | CUDA build-only, with the run command printed |
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
- Verified end-to-end on 2026-09-26 from a clean tree (slot-lifecycle
  target re-verified 2026-09-27): `make` builds all nine host binaries;
  `make test` → `PLAN TEST PASSED` + `ROPE TEST
  PASSED` + `FLAGS TEST PASSED` + `ENGINE SMOKE PASSED` + `ENGINE FEATURES
  PASSED` + `SLOT LIFECYCLE PASSED` + `PAR GOLDEN WRITTEN build/par_golden.bin` + `PARITY PASSED`
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

### device_gate_test — PASS (device gate, host; 2026-09-27)

- `mm_device_check` policy against `MM_TARGET_PRO4000`, no CUDA needed
  (pure: probed `mm_dev_info` + target profile). Wired into `make test`,
  the CMake `test` target and `scripts/run_host_tests.sh` (both build
  systems verified end-to-end 2026-09-27).
- **VRAM regression anchor:** the measured driver value on silicon
  (25,153,044,480 bytes = 23.4256 GiB, driver 595.71.05) must PASS —
  before the fix the gate rejected the target card itself and every GPU
  run needed `MIMFER_SOFT_DEVICE_GATE=1`. Boundary semantics: exactly at
  the 5%-below floor passes (inclusive), one byte below fails; above
  nominal passes (the gate is a lower bound).
- Name/CC/bandwidth: non-matching name rejected; cc 9.0 / 13.0 rejected,
  12.1 accepted; derived bandwidth at −10% passes (inclusive), above
  +10% and far-off nominal rejected.

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

### parity_test (GPU) — PASS (2026-09-27, RTX PRO 4000 Blackwell, sm_120)

- First full GPU parity comparison: `PARITY PASSED` — 1564 ops, 2108
  observables, 2,009,424 elements, 2,009,422 bit-exact. Class-1 (bf16):
  max 1.000 ulp on 2 elements (gate: 1). Class-2 (f32): max 0 ulp — the
  DeltaNet linear-attention state is bit-exact after the host-supplied
  decay fix (§6.6). First run with `MIMFER_SOFT_DEVICE_GATE=1` (the
  open VRAM-gate bug); re-run 2026-09-27 WITHOUT the bypass after the
  §6.8 fix: identical `PARITY PASSED` (2,009,422 bit-exact), hard gate
  `mm_device_check ok (target profile matched)`.
- Gate failures found and fixed on this run: KV block-id indexing IMA
  (§6.5) and the LIN decay constant (§6.6); the D2H scratch growth fix
  (§6.7) unblocked the readbacks.

### gpu_smoke (GPU) — PASS (2026-09-27, RTX PRO 4000 Blackwell, sm_120)

- `GPU SMOKE PASSED`: both hardcoded oracle heads match the host CPU
  reference (short: `[451 20 430 317 0 152 24 414]`, long:
  `[127 230 247 387 485 327 294 387]`), graph mode == direct mode,
  deterministic, clean device syncs.
- Re-run 2026-09-27 WITHOUT `MIMFER_SOFT_DEVICE_GATE` after the device-gate
  VRAM fix (§6.8): hard gate `mm_device_check ok (target profile matched)`
  + the same `GPU SMOKE PASSED` (both oracle heads, `rounds=17`/`rounds=71`).

### slot_lifecycle_test / gpu_slot — PASS (slot-lifecycle teardown; 2026-09-27, host + sm_120 silicon)

- One shared source (`tests/host/slot_lifecycle_test.c`), two builds: the
  host suite (`gcc`, CPU reference — where the kernels read the block
  table from the very mirror the test checks) and the GPU suite
  (`nvcc`, `MM_WITH_CUDA`; `make slot-lifecycle` / `make gpu-slot`,
  runbook §8).
- Proof structure: sequence A (40 prompt + 40 generated = 80 context
  tokens, spanning two 64-token KV blocks) runs to completion on slot 0;
  sequence B (5 + 12) then runs on the slot A freed and its full
  13-token stream must be byte-identical to B's stream on a fresh engine
  with the same seed.
- Checks: **S1** pool free-block count back to post-load after A AND
  after B (no KV leak; the mid-flight `free_at_65` probe confirms both
  blocks were actually allocated); **S2** reused-slot B stream ==
  fresh-engine B stream (the linear recurrence/conv state was reset);
  **S3** A's entire block-table row zeroed after teardown (the "no
  block" state).
- RED pre-fix on the current engine (S1 after A, S3 ×2, S1 after B, S2
  all FAIL); GREEN post-fix on host AND silicon (`SLOT LIFECYCLE PASSED`,
  `CUDA_VISIBLE_DEVICES=1`, hard device gate):
  `fresh B head=[409 2 419 419 419 419 235 419]`,
  `reuse A head=[14 452 433 189 266 207 201 74]` (`rounds=41`),
  `reuse B head=[409 2 419 419 419 419 235 419]` — the B heads are
  identical across CPU and GPU; deterministic across two GPU runs.
- See §6.9 for the bug it catches.

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

### 6.5 `src/kernels/cpu/cx.c` + `src/kernels/cuda/cx.cu` + `src/engine/engine.c` — KV block-id indexing — FIXED (2026-09-27)

- **Issue:** the block table holds block IDS (1..n_blocks, 0 = no
  block), but all four index sites used the id directly as a pool index
  (valid 0..n_blocks-1): CPU `kv_row`/`kv_row_w` and CUDA `k_kvstore`/
  `kv_row_dev`. With n_blocks=2 the first allocated block (id 2)
  indexed pool slot 2 — one past the last block.
- **Impact:** on the GPU, layer 7's V head-1 row (the last pool layer)
  was written 256 B past the KV pool into the `kv.tab` slot, corrupting
  `blk_tab_dev`; the next attention op then read a garbage block id →
  `illegal memory access` (the run-2/run-4 IMA). The fault was
  value-dependent: writing the loaded V data into the table slot faulted
  where a constant would not have. The CPU reference stayed self-
  consistent: the same stray write landed in a dead arena mirror while
  CPU reads used `blk_tab_host`, so the bug was invisible in the host
  suite and in the parity self-check.
- **Resolution:** `bid ? bid - 1 : 0` at all four sites. Values are
  unchanged (store and read were always self-consistent at the same
  wrong location) — only the pool layout moved, so `par_golden` was
  regenerated. Verified on sm_120: IMA gone, all 1564 ops run (previous
  runs aborted at prefill op81), parity passes.

### 6.6 `src/kernels/cuda/cx.cu` — `k_lin` decay constant — FIXED (2026-09-27)

- **Issue:** the first full GPU parity comparison (unblocked by §6.5)
  showed the DeltaNet f32 state deviating 5..256 f32 ulp from the golden
  (class-2 gate: 4 ulp) plus one downstream bf16 output 2 bf16 ulp off.
  `k_lin` computed `decay = expf(-0.1f)` in-kernel; device `expf`
  returns 0x3f67a36c where host glibc returns 0x3f67a36d (1 f32 ulp).
  Per-element libm gaps are covered by the 1 bf16 ulp gate (documented
  for the ROPE sinf/cosf), but a CONSTANT multiplied into every
  recurrence step compounds over the round sequence.
- **Resolution:** the launcher computes `expf(-0.1f)` with the same glibc
  call the CPU reference makes and passes it as a kernel argument — the
  recurrence is bit-identical on both paths (class-2 max deviation now
  0 ulp). `-fmad=false` (mandatory) plus identical loop order keep every
  other op bit-exact or within its gate.

### 6.7 `tests/cuda/parity_test.c` — D2H readback scratch — FIXED (2026-09-27)

- **Issue:** the CUDA-build readback scratch `g_dev` was never grown
  (only the host `d2h()` called `buf_grow`); the first copy ran with an
  undersized/NULL dst → `cudaMemcpyAsync` invalid argument (the run-2
  failure, silicon-confirmed `dst=(nil) class=unregistered`).
- **Resolution:** `d2h()` calls `buf_grow(&g_dev, &g_dev_cap, n)` before
  the copy (mirrors the host variant); the scratch persists for the run
  and keeps the max size seen — no per-readback realloc churn.

### 6.8 `src/config/config.c` — `mm_device_check()` VRAM gate — FIXED (2026-09-27)

- **Issue:** the device gate required `vram_bytes >= 24 GiB` nominal, but
  the driver-reported `totalGlobalMem` on the 24 GB RTX PRO 4000 is
  25,153,044,480 bytes = 23.4256 GiB — 2.39% below nominal (the
  driver/firmware reserves part of the card; verified 2026-09-27 with a
  driver-API probe, driver 595.71.05, both devices of the pair). The
  target card failed its own gate (`device gate: 23 GiB < required
  24 GiB`) and every GPU run needed the `MIMFER_SOFT_DEVICE_GATE=1`
  bypass. Not a profile mismatch and not a reporting bug: the driver
  total is simply the addressable (post-reserve) memory.
- **Resolution:** the gate now accepts up to `MM_VRAM_TOLERANCE_PCT` (5%)
  below the nominal `vram_bytes` (floor computed once in `size_t`,
  inclusive; no overflow at 64-bit). The constant + the measured anchor
  are documented in `include/mimfer/config.h`; the failure log now shows
  the tolerance and the floor. Regression: `tests/host/device_gate_test.c`
  (wired into `make test`, the CMake `test` target, and
  `scripts/run_host_tests.sh`): measured value passes, floor boundary is
  inclusive, floor−1 / wrong name / wrong cc / off-nominal bandwidth are
  rejected.
- **Silicon verification (2026-09-27, WITHOUT the bypass):**
  `CUDA_VISIBLE_DEVICES=1 build/gpu_smoke` → `mm_device_check ok (target
  profile matched)` + `GPU SMOKE PASSED` (both oracle heads,
  `rounds=17`/`rounds=71`); `build/parity_test build/par_golden.bin` →
  `PARITY PASSED` (2,009,422/2,009,424 bit-exact — identical to the
  bypassed run). Probe line: `NVIDIA RTX PRO 4000 Blackwell (sm_120, 70
  SMs, 23 GiB, 672 GB/s derived, l2 48 MiB, smem/SM 100 KiB)` — derived
  bandwidth exactly 672 GB/s (the ESTIMATE held); L2 reports 48 MiB vs
  the 96 MiB ESTIMATE (soft check, log-only, never fatal — keep watching
  under production load).
- **Docs updated:** GPU_VALIDATION.md (header state, §3.2, §3.3 item 4,
  troubleshooting), VALIDATION_CHECKLIST.md (Gate 0 suite list),
  RELEASE_READINESS.md (R4, A5 — closed), README.md (hardware scope),
  Makefile + CMakeLists.txt + `scripts/run_host_tests.sh` (new test
  stage). Committed on `bugfix/vram-gate-tolerance`.

### 6.9 `src/engine/engine.c` — `step_decode()` — slot teardown missing on sequence completion — FIXED (2026-09-27)

- **Issue:** the round scheduler recycles slots: when a sequence hit its
  budget or the context cap, `step_decode` only flipped the scheduler
  state (`mm_sched_done`) and immediately recycled the slot for the next
  sequence — without tearing down the finished sequence's runtime state.
  Three defects: **S1** the slot's KV blocks were never released
  (`mm_kvpool_release_slot` had no callers — a KV leak per sequence,
  pool exhaustion under multi-request load); **S2** the slot's linear
  recurrence/conv state (48 Gated-DeltaNet layers + conv tails) carried
  the finished sequence's values into the next sequence's forward pass
  (`mm_linstate_reset` had no callers); **S3** the device block-table
  row kept the finished sequence's block ids until the next prefill's
  incremental push overwrote them — breaking the pool's documented
  "zero row == no block" invariant (and, on the CUDA build, leaving
  stale ids readable by the kernels between prefill and the next write).
- **Impact:** the second sequence on a reused slot computed with the
  first sequence's linear state and a stale block table; on the tiny
  shape the two B streams diverged mid-generation (S2 FAIL in the
  regression); the pool free count kept dropping (S1 FAIL).
- **Resolution:** `teardown_slot()` in the `step_decode` completion
  branch, at the round boundary after the decode round's token readback
  host sync (the CUDA `fetch_tokens` sync has already quiesced both
  streams — the reset on the compute stream and the row push on the
  xfer stream need no event ordering; the next round's `push_ctrl`
  re-establishes the usual ordering): `mm_kvpool_release_slot` (frees
  the blocks, zeroes the host row) → `mm_linstate_reset` →
  `mm_kvpool_push_tab` (the zeroed row; device sync on the CUDA build,
  host→host copy on the reference build).
- **Verified:** `slot_lifecycle_test` / `gpu_slot` (one shared source,
  A = 40+40 tokens spanning two KV blocks, then B on the freed slot):
  RED pre-fix (S1 after A, S3 ×2, S1 after B, S2 all FAIL), GREEN
  post-fix on host AND sm_120 silicon — `SLOT LIFECYCLE PASSED`
  (reused-slot B byte-identical to the fresh-engine B and to the host
  suite; pool free count back to post-load after A and B; block-table
  row zeroed; deterministic across two GPU runs). Full host suite green
  including the bit-exact parity self-check (the single-sequence golden
  path is untouched — teardown only runs at completion). Committed on
  `bugfix/slot-lifecycle` (3 commits: red test, fix, runbook/docs).

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

**Completed 2026-09-27 (this session):** GPU execution VERIFIED on
silicon (RTX PRO 4000 Blackwell, sm_120, CUDA 13.1.115, driver
595.71.05, run directly from vibe-workspace): both run-2 blockers
cleared (NULL `g_dev` dst → `buf_grow` fix, §6.7; graph-mode IMA →
1-based KV block-id indexing bug, §6.5 — not a graph bug) plus the
newly-exposed LIN decay gate failure (§6.6); temporary IMA/D2H
instrumentation removed. `PARITY PASSED` (1564 ops, 2,009,424
elements, 2,009,422 bit-exact; f32 state bit-exact) + `GPU SMOKE
PASSED` (graph == direct, oracle heads); host suite green; `par_golden`
regenerated (pool layout moved, values and sampled heads unchanged).
Committed on branch `bugfix/kv-index-ima` (3 commits: KV index fix,
decay fix, instrumentation cleanup + D2H scratch).

**Completed 2026-09-27 (slot-lifecycle):** the P0 slot-teardown defects
(S1/S2/S3, §6.9) — the scheduler recycled a finished sequence's slot
without releasing its KV blocks, resetting its linear/conv state, or
zeroing its block-table row. Shipped as `slot_lifecycle_test`
(regression, RED pre-fix, one shared host/GPU source) + `teardown_slot()`
in the `step_decode` completion branch + runbook §8 / checklist Gate 2.5.
GREEN on host and silicon (`SLOT LIFECYCLE PASSED`; reused-slot B
byte-identical to a fresh engine on CPU and GPU; deterministic across
two GPU runs; full host suite incl. the bit-exact parity self-check
green). Committed on `bugfix/slot-lifecycle` (3 commits: red test, fix,
runbook/docs).

**NEXT TASK: merge the three bugfix branches, then the pinned-ctrl H2D cleanup**
(1) `bugfix/kv-index-ima` (KV index fix, decay fix, instrumentation
cleanup + D2H scratch), `bugfix/vram-gate-tolerance` (device-gate VRAM
tolerance + `device_gate_test` regression) and `bugfix/slot-lifecycle`
(regression + S1/S2/S3 teardown fix + runbook/docs) hold the verified
commits — merge per the branch rules (the repo has no `develop` branch
yet; merging to `main` is a human gate); (2) DONE — the device-gate VRAM
bug (driver reports 23.4256 GiB < 24 GiB nominal on the 24 GB card):
fixed with the 5% tolerance (`MM_VRAM_TOLERANCE_PCT`) + regression test,
and the target card now passes the hard gate natively on silicon (no
`MIMFER_SOFT_DEVICE_GATE`, §6.8); (3) DONE — the slot-lifecycle teardown
bug (S1/S2/S3): fixed with `teardown_slot()` at sequence completion
(§6.9), regression-verified on host + silicon; (4) the pinned-block ctrl
design: `e->ctrl_dev` points into the pinned host block yet `push_ctrl`
issues an H2D `cudaMemcpyAsync` into it (kernels read it zero-copy) —
driver-tolerated; make it an explicit host memcpy or a real device
buffer; (5) then Blackwell optimizations (FMA / tensor-core paths) — each
MUST re-pass the parity test (§4.6); (6) optional `compute-sanitizer`
sweep once a toolkit that includes it is available.

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

Order (full commands in GPU_VALIDATION.md §5–§9) — COMPLETED 2026-09-27 (steps 1–3b passed on silicon, incl. the new `gpu_slot` slot-lifecycle teardown gate; the sanitizer step unavailable — no compute-sanitizer in the local toolkit; optimizations = the next task):

1. Golden + oracle pre-gate on the GPU machine (host gcc):
   `par_golden` → `PAR GOLDEN WRITTEN`; `par_selfcheck` → `PARITY PASSED`
   (if the self-check fails there, rebuild the golden on that machine —
   the golden is CPU-written and must match the validating machine).
2. `parity_test` (nvcc, `-fmad=false -DMM_WITH_CUDA -arch=sm_120`) →
   expect `PARITY PASSED` (1 bf16 ulp gate on computed ops).
3. `gpu_smoke` (nvcc) → expect `GPU SMOKE PASSED` (graph == direct,
   oracle heads, deterministic, clean device syncs).
3b. `gpu_slot` (nvcc; one shared source with the host
   `slot_lifecycle_test`) → expect `SLOT LIFECYCLE PASSED` (reused-slot
   B byte-identical to the fresh-engine B; pool free count back to
   post-load after A and B; block-table row zeroed) — VERIFIED 2026-09-27
   on silicon, deterministic across two runs.
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
| Project health | GREEN (host + GPU parity verified 2026-09-27) |
| Host correctness | VERIFIED |
| Memory safety | VERIFIED |
| Determinism | VERIFIED |
| Planner | COMPLETE |
| CPU path | COMPLETE |
| CUDA runtime + kernels | COMPILED (real nvcc 13.1.115, driverless, 2026-09-26) — runtime layer, device mem, cx.cu launch set; sm_120 fatbin inspected via cuobjdump (all 14 kernels present, zero warnings); graph-capture-legal, round-boundary I/O wired; SILICON-VERIFIED 2026-09-27 (parity PASSED + smoke PASSED on sm_120) |
| Parity tooling | COMPLETE + host-VERIFIED (par_golden golden writer; parity_test dual-buildable; self-check passes bit-exact) |
| GPU smoke test | GPU-VERIFIED 2026-09-27 (graph vs direct, oracle heads match the host reference, reproducible tokens) |
| Validation docs | COMPLETE — GPU_VALIDATION.md (runbook: requirements, procedure, troubleshooting, report template; all link sets include `src/rope/rope.c`), VALIDATION_CHECKLIST.md (go/no-go; Gate 0 requires `make test` green), RELEASE_READINESS.md (status/risks/audit incl. the new feature rows; R4 VRAM defect fixed + A5 gate-pass verified 2026-09-27), docs/dflash2.md (declared speculative-decoding scope); 2026-09-26 (gate sections re-synced 2026-09-27) |
| README | COMPLETE — README.md: Current Model Support (two NInfer Qwen3.8-27B reference models only), Current Artifact Support (NInfer Artifact V2/V3), Hardware Scope (RTX PRO 4000 Blackwell only; soft-gate is a testing affordance), **Engine & CLI Feature Surface** (flag-by-flag status table: wired / validated scaffolding / validated hook), Extensibility (suckless-inspired, architecture influence not code dependency); 2026-09-26 |
| Feature surface | COMPLETE (host) — RoPE tables (`src/rope/rope.c`, plain + YaRN; plain bit-identical to legacy formula so the golden is unchanged), CLI flag grammar (`src/flags/flags.c`), weights profiles + `mm_engine_cfg_validate` cross-field rules (`src/config/`), `--spec`/`--vision` gates in `mm_engine_load`; tests: `rope_test`, `flags_test`, `device_gate_test` (device-gate policy; VRAM tolerance regression-anchored to the measured 23.4256 GiB), `engine_features_test` (e2e YaRN q-buffer difference vs plain, plain byte-identical, spec/vision refusals); declared speculative scope: docs/dflash2.md |
| Governance | COMPLETE — .clinerules (mandatory rules: memory, branch, architecture, scope, CUDA parity-first, code quality, documentation, philosophy), .gitignore (build/CUDA/test/editor artifacts ignored; all docs + .clinerules explicitly kept versioned; root Makefile + tests/ un-ignored), git repo with 3 commits on `main`; work branch `feature/cmake-flag-surface` pushed to origin (CMake now exists in the tree — `CMakeLists.txt`, host parity + optional CUDA — so the branch name is no longer a historical artifact; see the Build system row); 2026-09-26; 2026-09-27: `bugfix/kv-index-ima` fast-forward merged into `main` (tip cf473e3), `bugfix/vram-gate-tolerance` branch created for the device-gate VRAM fix + `device_gate_test` (pending merge) |
| Build system | COMPLETE — root `Makefile` (GNU Make only): `make`/`make host` (8 host bins in `build/`), `make test` (7-stage sequential host suite, fail-fast: plan → rope → flags → device-gate → engine-smoke → engine-features → parity-selfcheck), `make cuda`/`gpu-parity`/`gpu-smoke` (build-only + printed run commands), `check-nvcc`, `help`, `clean`, `distclean`; `CUDA_ARCH ?= sm_120`; `tests/host/par_selfcheck.c` wrapper to the shared comparator; two GNU Make 4.3 traps fixed (order-only dir prerequisite skip; default goal = first file target — see §4.7); verified end-to-end 2026-09-26; **B1 fix applied 2026-09-26**: GPU recipes now link `src/kernels/cuda/cx.cu` (via `GPU_CU_SRCS`, no `-x c`) — verified by `make -n` expansion; **CMake parity 2026-09-26**: `CMakeLists.txt` — same 8 host binaries via CMake/Ninja, Makefile-identical host flags (`-std=c11 -Wall -Wextra -Werror`, no host `-O*`/`-DNDEBUG`/`gnu11`), `test`/`golden`/`smoke`/`parity` CMake targets green; optional CUDA: `-DMIMFER_ENABLE_CUDA=ON` (+ `-DMIMFER_NVCC_EXTRA` for driverless stub linking), CMake default stays host-only, `--target cuda` builds the GPU binaries (mirrors `make` default / `make cuda`); **nvcc `-x` fix 2026-09-26**: all `-x` flags removed from Makefile + CMake + documented recipes — CUDA 13.1's `-x` is a *global* last-value-wins option (per-source `-x c` had been compiling `cx.cu` as plain C), language is now by file extension; `extern "C"` guards added to 17 public headers for the C++ TU; `make cuda NVCC_EXTRA=-L<toolkit>/lib64/stubs` and the CMake CUDA build verified against the **real nvcc 13.1.115** (zero warnings; fatbin: 14 kernels, sm_120); host build stays hermetic (no CUDA toolkit needed for the default target, suite green) |
| Final build audit | COMPLETE (2026-09-26) — `BUILD_AUDIT.md` (B1 critical GPU-link fix + B10 critical: GPU `parity_test` had no `kx_cpu_oppref` provider; fixed with a `#ifdef MM_WITH_CUDA` reference copy in `tests/cuda/parity_test.c` — host recipe byte-identical, `make test` re-green; B2–B9 documented: 3 unregistered TUs pass `-fsyntax-only` under release flags, no CMake exists, `HDRS` caveat, tmux junk, stale README sentence fixed), `SOURCE_TREE.md` (all 57 tracked files: role/size/registration), `MODULE_DEPENDENCIES.md` (per-TU include graph, module map, cross-TU symbol deps, `MM_WITH_CUDA` split); no missing headers; no dead sources; Make = sole build system and matches the documented verbatim commands |
| GPU execution | RUN 2 (2026-09-26, RTX PRO 4000 Blackwell, CUDA 13.2, driver 595.71.05): engine create+load PASS — run-1's opacity blockers (inverted `mm_log` filter; unlogged failure paths) are fixed and held. `mm_device_check` gate now visibly fails on VRAM: `device gate: 23 GiB < required 24 GiB` — SEPARATE known bug, deferred; `MIMFER_SOFT_DEVICE_GATE=1` bypasses it. New failure: first D2H observable readback — `E cuda_rt.c:113` (CK in `mm_d2h_async`) + `E parity_test.c:322` + `E parity_test.c:528`: `cudaMemcpyAsync(dst, src, n, cudaMemcpyDeviceToHost, g_st[st])` → invalid argument. ROOT CAUSE (static analysis, high confidence): the D2H readback scratch `g_dev` (parity_test.c:296) is never allocated in the CUDA build — only the host `d2h()` calls `buf_grow()` — so the first copy passes dst=NULL. D2H path instrumented (temporary, remove after fix): `mm_d2h_async` (cuda_rt.c) logs dst/src/bytes/raw-stream+id + `cudaPointerGetAttributes` class per endpoint before every copy; the three call sites (per-op observable, pool dump, `fetch_tok`) log readback context. CUDA 13.1 C-API nits fixed in the instrumentation: `struct cudaPointerAttributes` (no C typedef), runtime `cudaMemoryType` has no `Unifiable` member. Re-verified: `make test` green; `make gpu-parity` + `make gpu-smoke` from clean with real nvcc 13.1.115, zero warnings. RUNS 3–4 (2026-09-27, same hardware, executed DIRECTLY FROM vibe-workspace — no ai-dev needed): root cause SILICON-CONFIRMED — rebuilt `build/parity_test build/par_golden.bin` (fresh local nvcc build, direct dispatch) printed `W cuda_rt.c:136: d2h_async: dst=(nil) class=unregistered src=... class=device bytes=512` before the `invalid argument` (two stable runs). Direct-dispatch kernels (k_embed + prefill ops) launch AND sync clean on real Blackwell (70 SMs, 23.43 GiB card, 672 GB/s). NEW BUG: `gpu_smoke` graph mode (no_graph=0) — first graph-launch stream sync → `E cx.cu:90: stream sync: an illegal memory access was encountered` (mm_kx_invoke, prefill); fault taints the context (every later `cudaStreamCreateWithFlags` fails with the same IMA); direct dispatch does NOT IMA → graph capture/replay path suspected. Environment: CUDA dev0 = PCI 0000:05:00.0 (/dev/nvidia1) BUSY (ctx create → OOM, VRAM held by an external workload, presumably ai-dev); CUDA dev1 = PCI 0000:06:00.0 (/dev/nvidia0) FREE (23.43 GiB total, ~23.2 GiB free) → always `CUDA_VISIBLE_DEVICES=1`; userspace driver libs 595.91.07 vs kernel 595.71.05 — nvidia-smi NVML mismatch only, CUDA driver API fully functional (probe: cuInit, enumeration, context, H2D/D2H round-trip verified); local toolkit /home/razvijalec/tools/cuda/usr/local/cuda-13.1 (real nvcc V13.1.115, NO compute-sanitizer); `make cuda NVCC=<that path>/nvcc` builds both GPU binaries from clean. NEXT: one-line `buf_grow` fix in the CUDA `d2h()` (separate commit), then strip instrumentation, then local re-run. ALL DONE 2026-09-27 — RUN 5: `PARITY PASSED` (1564 ops, 2,009,424 elements, 2,009,422 bit-exact; f32 state bit-exact) + `GPU SMOKE PASSED` (graph+direct); the "graph-mode IMA" was not a graph bug (1-based KV block-id indexing, §6.5) and the fix-unblocked decay gate failure (§6.6) was fixed host-side; `par_golden` regenerated; temporary instrumentation removed; committed on `bugfix/kv-index-ima` (3 commits) |
| Recommended next step | (1) merge `bugfix/kv-index-ima` (3 verified commits) + `bugfix/vram-gate-tolerance` (device-gate VRAM fix + `device_gate_test`) + `bugfix/slot-lifecycle` (slot-teardown regression S1/S2/S3 + `teardown_slot()` fix + runbook) per the branch rules — human gate for main (no `develop` branch exists yet); (2) DONE — device-gate VRAM bug fixed 2026-09-27 (5% tolerance `MM_VRAM_TOLERANCE_PCT` + `device_gate_test`; silicon re-run WITHOUT the bypass: hard gate `mm_device_check ok`, `GPU SMOKE PASSED` + `PARITY PASSED`; runbook updated); (3) DONE — slot-lifecycle teardown fixed 2026-09-27 (`teardown_slot()` at sequence completion: KV release + lin/conv reset + zeroed block-table row; `slot_lifecycle_test`/`gpu_slot` RED pre-fix, GREEN on host + silicon — `SLOT LIFECYCLE PASSED`, reused-slot B byte-identical to a fresh engine, deterministic across two GPU runs; runbook §8 / checklist Gate 2.5 added); (4) the pinned-ctrl H2D design (H2D into a pinned host dst — driver-tolerated; make it explicit); (5) Blackwell optimizations (FMA / tensor-core) — each MUST re-pass the parity test; (6) `compute-sanitizer` sweep once a toolkit with it is available |
| Audit findings | Final build audit 2026-09-26 (`BUILD_AUDIT.md` B1–B10): B1+B10 GPU-link defects FIXED (see above); artifact/tokenizer/telemetry written but unintegrated — all three pass `-fsyntax-only` under the exact release flags (not latent breakage); **no missing headers, no dead sources** (every TU registered, or the 3 above); zero TODO/FIXME markers; ~1.3 GB tmux logs at repo root (junk, R6, human-decision); docs/architecture.md + docs/design.md referenced but absent; README stale Extensibility sentence fixed (B9); README docs table + RELEASE_READINESS §7 synced to the three new audit docs |
| Risk | Compile/link layer is CLOSED (real nvcc 13.1.115, both build systems, zero warnings, fatbin verified). Silicon: run 1 failed at `mm_engine_create` (root cause: inverted `mm_log` filter — FIXED, kept permanently); run 2 (soft device gate) passes create+load and fails at the first D2H readback — root cause identified statically (NULL `g_dev` dst; the pre-copy instrumentation will confirm on silicon), one-line fix pending. OPEN BUGS: the pinned-block ctrl design — `e->ctrl_dev` points into the pinned host block yet `push_ctrl` issues an H2D `cudaMemcpyAsync` into it (the kernels read `e->ctrl_dev` directly, zero-copy); run 2's H2D calls returned success, but that is driver-dependent behavior for a host dst + H2D kind — make it an explicit host memcpy or a real device buffer once the D2H path is cleared. Temporary instrumentation REMOVED 2026-09-27 (committed on `bugfix/kv-index-ima`); the run-2 IMA root cause (KV block-id indexing) and the LIN decay gate failure are fixed and silicon-verified. RESOLVED 2026-09-27: the device-gate VRAM bug — the driver reports 23.4256 GiB (2.39% below nominal) on the 24 GB card (driver/firmware reserve, measured with a driver-API probe); the gate now tolerates up to 5% below nominal (`MM_VRAM_TOLERANCE_PCT`) + `device_gate_test` regression; silicon re-run WITHOUT `MIMFER_SOFT_DEVICE_GATE` passed the hard gate natively (`mm_device_check ok (target profile matched)`, `GPU SMOKE PASSED` + `PARITY PASSED`; §6.8). Remaining risks: (1) the pinned-ctrl H2D design, (2) optimization passes (each must re-pass parity) |
