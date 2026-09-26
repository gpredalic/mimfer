# RELEASE_READINESS.md — Status & Risk Summary (mimfer)

> Snapshot: 2026-09-26. Scope: what is complete, what is pending, what is
> risky, what has not been verified. Companions: `READ_MEMORY.md` (state),
> `GPU_VALIDATION.md` (how to validate on a GPU), `VALIDATION_CHECKLIST.md`
> (go/no-go gate). All findings below were taken from the source tree on
> this date; no speculative items are included.

## 1. Verdict

- **Host (CPU) reference: release-ready.** Planner, CPU kernels, engine,
  KV pool, scheduler, sampling — all implemented and verified (unit +
  smoke + ASan/UBSan), deterministic across runs.
- **GPU path: code-complete, not hardware-verified.** `src/cuda/` and
  `src/kernels/cuda/cx.cu` have never been compiled with nvcc or executed
  on silicon. No GPU claim may be made until
  `VALIDATION_CHECKLIST.md` passes on real hardware.
- **Known deferred scope** (by design, documented in-tree): artifact
  loading, tokenizer, telemetry, FP8 KV, MTP draft — written or declared,
  not integrated (§3).

## 2. Completed subsystems (with verification evidence)

| Subsystem | Location | Status | Evidence |
|-----------|----------|--------|----------|
| Planner (op-graph + activation planning) | `src/plan/plan.c` | COMPLETE | `plan_test` (40 checks, real plan.c/alloc/config/mimfer) |
| CPU reference kernels (all 18 executable opcodes) | `src/kernels/cpu/cx.c` | COMPLETE | `engine_smoke`, golden writer, ASan/UBSan clean |
| Kernel dispatcher (C ABI) | `src/kernels/kx.c/.h` | COMPLETE | linkage guards verified; all host tests green under `-Werror` |
| Tensor registry | `src/model/tensor_registry.c` | COMPLETE | linked into all engine builds |
| Engine lifecycle (create/load/step/destroy) | `src/engine/engine.c` | COMPLETE (host) | `engine_smoke` 4-case byte-identical token streams, 2 runs |
| KV paged pool | `src/kv/kv.c` | COMPLETE | off-by-one heap overflow fixed + ASan-verified |
| Scheduler (rounds, slots) | `src/sched/sched.c` | COMPLETE | exercised by `engine_smoke`/`gpu_smoke` paths |
| Sampling (greedy + seeded non-greedy) | `src/sampling/sampling.c` | COMPLETE | deterministic across runs (same seed) |
| CUDA runtime layer (streams/events/graphs/error wrappers, host no-op stubs) | `src/cuda/cuda_rt.c` | WRITTEN | host stubs exercised by every host build; CUDA side unexecuted |
| Device arenas + pinned I/O | `src/alloc/alloc.c`, `src/cuda/cuda_mem.c` | WRITTEN | host paths exercised; device paths unexecuted |
| CUDA kernels (parity-first mirrors of the CPU reference) | `src/kernels/cuda/cx.cu` | WRITTEN | syntax-checked only; **never compiled with nvcc** |
| Parity tooling (golden writer + dual-mode parity test + host self-check) | `tests/host/par_golden.c`, `tests/cuda/parity_test.c` | COMPLETE (host side) | self-check `PARITY PASSED`, bit-exact against golden |
| RoPE frequency tables (plain + YaRN NTK-by-parts) | `src/rope/rope.c`, `include/mimfer/rope.h` | COMPLETE (host) | `rope_test` (plain bit-exact vs legacy formula; YaRN vs transformers v4.45.0 reference formula); consumed by CPU+CUDA RoPE kernels via the round control buffer; plain path leaves the golden bit-identical |
| CLI flag surface (one grammar table + cross-field validation) | `src/flags/flags.c`, `include/mimfer/flags.h` | COMPLETE (host) | `flags_test`: every flag parsed/range-checked, malformed input rejected with explicit errors, `--help` lists all flags; YaRN factor + `--spec`/`--draft-tokens`/`--lm-head-draft` cross-field rules via `mm_engine_cfg_validate` |
| Weights profiles (quasar / neroued context defaults + artifact metadata) | `src/config/config.c` (`MM_PROFILES`) | COMPLETE (host) | profile lookup + "explicit flags win" defaulting covered by `flags_test`/`engine_features_test`; the profiles are metadata (max ctx, chunk, DFlash2 window ceiling, MTP layers) — the artifact loader itself is still unintegrated (§3) |
| Engine feature gates (spec backend + vision hook) | `src/engine/engine.c` | COMPLETE (host) | `engine_features_test`: `--spec mtp/dflash2` refused at start with `MM_ERR_UNSUPPORTED` + explicit log; vision hook validates then refuses image submission; declared scope in `docs/dflash2.md` |
| Feature test harness (engine-level, host) | `tests/host/engine_features_test.c` | COMPLETE | one prefill round: YaRN table demonstrably changes the rotated q buffer vs plain (the table reaches the kernels); plain runs byte-identical (golden path unchanged) |
| End-to-end GPU smoke test | `tests/cuda/gpu_smoke.c` | WRITTEN | written + syntax-checked; pending GPU execution |
| Device profile + gate (RTX PRO 4000 target, soft-gate env var) | `src/config/config.c`, `src/engine/engine.c` | WRITTEN | host-verified (gate code path only reached in CUDA builds) |

## 3. Pending / unintegrated subsystems

| Item | State | Notes |
|------|-------|-------|
| Artifact loading (`.mimfer`) | `src/artifact/artifact.c` written, **not in any documented build, not called** | engine stands in the tiny shape instead (`engine.c:377`); struct field `mm_engine::art` exists but is unused in the lifecycle. Not verified by any test. |
| Tokenizer | `src/tokenizer/tokenizer.c` written, **not in any documented build, not called** | same status; header included by `engine.h` only for the struct field. |
| Telemetry | `src/telemetry/telemetry.c` written, **not in any documented build, not called** | same status. |
| FP8 KV cache (`MM_KV_FP8`) | declared (`config.h`, marked "phase 2"); pool strides handle it (`kv.c`); kernels write bf16 only | intentional phase-2 scope. |
| MTP draft / DFlash2 speculative decoding (`cfg.spec`, `cfg.draft`, `--spec`, `--draft-tokens`, `--lm-head-draft`) | flags parsed + cross-validated; profile metadata carries the draft ceilings; **non-off backends refused at engine start** (`MM_ERR_UNSUPPORTED`); no draft/verify flow in the step loop | declared scope, unimplemented flow — declared scope documented in `docs/dflash2.md` |
| Blackwell optimizations (FMA/tensor-core paths) | not started | explicitly blocked until hardware validation (parity gate must re-pass after any such change). |
| `docs/architecture.md`, `docs/design.md` | **do not exist** | referenced from comments in `config.h`, `cuda_rt.h`, `tensor.h`; the documented design lives in `READ_MEMORY.md` instead. (`docs/dflash2.md` — the speculative-decoding declared scope — does exist; created 2026-09-26.) |

## 4. Known risks

| # | Risk | Likelihood / impact | Mitigation / status |
|---|------|--------------------|---------------------|
| R1 | **`cx.cu` has never been compiled.** First nvcc run may surface C++/CUDA-specific compile errors in the only uncompiled TU. | Medium / blocks everything downstream | Isolated by build order in `GPU_VALIDATION.md`; all 14 other TUs are verified. Failure mode is a build error, not silent misbehavior. |
| R2 | First-run driver/toolkit interaction: launch attributes on graph nodes, thread-local stream capture, L2 access-policy windows on the real 12.8/13.x driver. | Low-Medium | L2 window node rejection is soft by design (warning, continue); capture is a single documented path (greedy, at load). Any capture error is treated as a real bug (§9.2 of the runbook). |
| R3 | **Golden is compiler/machine-dependent** (FP behavior of the host compiler that wrote it). A golden built on one machine can mismatch another's CPU reference. | Medium / would corrupt attribution of parity failures | Oracle pre-gate: `par_selfcheck` must pass on the validating machine before any GPU comparison; golden rebuild rule documented (`GPU_VALIDATION.md` §5.3). |
| R4 | Device-profile fields marked **ESTIMATE** (bandwidth 672 GB/s, L2 96 MiB, smem 100 KiB) — the derived-bandwidth check (±10%) can gate the *real* PRO 4000 if driver-reported clocks/bus differ. | Medium on the target card | Probed values logged at startup; `MIMFER_SOFT_DEVICE_GATE` bypass; the report template captures the probed line. |
| R5 | L2 window is v1: one window on **all** kernel nodes of each graph; per-layer windows come with the per-layer plan split (documented in `cuda_rt.c`). | Low (optimization only) | Node-level rejection is non-fatal. |
| R6 | **Repo hygiene:** `tmux-client-153455.log` (18 KB), `tmux-out-153457.log` (64 MB), `tmux-server-153457.log` (**6.4 GB**) at the repo root — session junk, not project files. | Low (disk space, confusing distribution) | Recommend deletion (human decision). |
| R7 | Working copy has **no VCS** (no `.git`); history exists only via `READ_MEMORY.md`. | Low-Medium | Copy the tree before GPU validation; consider git init with the docs as the first commit. |

## 5. Unverified assumptions

| # | Assumption | Where it matters |
|---|------------|------------------|
| A1 | All CUDA API usage (`cudaGraph*`, launch attributes, thread-local capture, access-policy windows) is legal on the 12.8/13.x driver as invoked. | first `gpu_smoke`/`parity_test` run |
| A2 | 1-bf16-ulp parity holds on silicon with `-fmad=false` (verified on host only: FMA-free oracle vs FMA-free oracle). | `parity_test` gate |
| A3 | The op sequence is graph-capture-legal in thread-local mode (the launchers skip syncs/readbacks while `mm_cuda_capturing()`). | graph-mode load |
| A4 | Greedy GPU sampling is deterministic across runs and across graph/direct modes. | `gpu_smoke` determinism checks |
| A5 | The device gate passes on the real PRO 4000 (bandwidth derivation depends on driver-reported memory clock × bus width). | engine create on the target card |
| A6 | Tiny-shape VRAM headroom (single-digit MiB) on any 24 GiB+ card. | trivially expected; unmeasured |
| A7 | Arena/host-mirror sizes are correct on first device allocation (`cudaMalloc` once per arena at init). | first CUDA load |

## 6. Hardware validation status

**Verified on host (this machine, 2026-09-25/26):**
`plan_test` PASS · `rope_test` PASS (plain bit-exact vs legacy formula;
YaRN vs reference formula) · `flags_test` PASS (full grammar, cross-field
rules, explicit errors) · `engine_smoke` PASS (4 cases, byte-identical
token streams across two runs) · `engine_features_test` PASS (YaRN table
reaches the kernels; plain byte-identical; spec/vision refusals) ·
`par_golden` written (3,761,596 B, reproducible
byte-for-byte on re-run) ·
`par_selfcheck` PASS (bit-exact, all classes at ulp 0) · ASan+UBSan clean
(no leaks/overflow/UB) · host regression green under `-Werror` (now the
`make test` suite, 6 binaries, strictly sequential).

**Not verified (requires a GPU; exactly what remains):**
1. `/tmp/par_golden` build+run on the GPU machine (host gcc) → `PAR GOLDEN WRITTEN`
2. `/tmp/par_selfcheck` build+run (host gcc) → `PARITY PASSED` (oracle gate)
3. `/tmp/parity_test` build (nvcc) + run → `PARITY PASSED`
4. `/tmp/gpu_smoke` build (nvcc) + run → `GPU SMOKE PASSED`
5. optional: `compute-sanitizer` sweep → 0 errors

All commands, expected outputs, and failure interpretation:
`GPU_VALIDATION.md`. Go/no-go: `VALIDATION_CHECKLIST.md`.

## 7. TODO / placeholder audit (actual findings only)

- **`TODO` / `FIXME` / `HACK` / `XXX` markers: zero** in `src/`, `include/`,
  `tests/`.
- **Intentional stubs (by design, documented in-tree):**
  - `src/cuda/cuda_rt.c` host no-op stubs (graph ops return
    `MM_ERR_UNSUPPORTED`; H2D/D2H/mset are real synchronous host copies) —
    this is how the host reference build works, not unfinished code.
  - `tests/host/plan_test.c` local stubs for `mm_kx_invoke` /
    `mm_tens_find` and a 4 MiB "dummy" weight buffer — planning-only test,
    never executed; documented in the test header.
  - `include/mimfer/cuda_mem.h` host stubs: plain `calloc`/`free` (honestly
    noted as NOT pinned; host-only).
- **`MM_ERR_UNSUPPORTED`** (`mimfer.h`): a legitimate error code ("feature
  not implemented on this build"), used by the host stubs.
- **`#error` directives: exactly one** — `tests/cuda/gpu_smoke.c:51`
  (intentional guard requiring `-DMM_WITH_CUDA`).
- **No `abort()` / `exit()` / `__builtin_trap` in library code.**
- **Written-but-unintegrated modules:** `artifact.c`, `tokenizer.c`,
  `telemetry.c` (§3) — self-contained, but in no documented build and
  untested.
- **Declared-but-unimplemented:** `MM_KV_FP8` ("phase 2"), MTP `draft` /
  DFlash2 speculative-decoding flow (§3; declared scope in
  `docs/dflash2.md`, non-off backends refused at engine start).
- **Missing referenced docs:** `docs/architecture.md`, `docs/design.md`
  (referenced from three headers; still absent). `docs/dflash2.md` **was**
  referenced by `flags.h`/`config.h`/`engine.c` and **did not exist** before
  2026-09-26 — it is now created and is the declared scope of the
  speculative surface.
- **Junk files at repo root:** three `tmux-*.log` files, total ≈ 6.4 GB (R6).
- **Documentation gaps found in the audit:** the documented nvcc commands in
  `READ_MEMORY.md` §4.6 carried no `-arch` flag; `GPU_VALIDATION.md` §6/§7
  add `-arch=sm_120` (replace per card). The verbatim `gcc`/`nvcc` recipes in
  `GPU_VALIDATION.md` §5–§7 and `READ_MEMORY.md` §4.6 now include the RoPE
  module (`src/rope/rope.c`) required by the engine link. No other
  undocumented build assumptions were found (flags, include paths, and env
  vars are now fully listed in `GPU_VALIDATION.md` §3).

## 8. Release gates

| Claim | Gate | Status |
|-------|------|--------|
| Host reference correctness | host regression + sanitizers (done) | **PASS (verified)** |
| Determinism (CPU) | two-run byte-identity (done) | **PASS (verified)** |
| GPU numeric parity | `VALIDATION_CHECKLIST.md` Gate 1 on real silicon | **PENDING** |
| GPU end-to-end behavior (graph/direct/determinism) | Gate 2 on real silicon | **PENDING** |
| GPU memory safety | Gate 3 (sanitizer) on real silicon | **PENDING (optional but recommended)** |
| GPU performance / Blackwell optimization | after all gates pass; each optimization must re-pass the parity gate | **BLOCKED** (no hardware access) |
