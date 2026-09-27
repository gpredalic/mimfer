# GPU_VALIDATION.md — Hardware Validation Runbook (mimfer)

> **Audience:** a human developer with a real NVIDIA GPU, performing the
> first GPU validation of the mimfer CUDA path.
>
> **Scope:** this document is the complete, self-contained procedure. The
> host (no-GPU) side of the project is fully verified (planner, CPU
> reference, engine smoke, sanitizers). Everything below is what must be
> done on the GPU machine. If a step fails, use §9 (Troubleshooting) and
> file a report per §10.
>
> **State of the code:** the CPU reference is the correctness oracle and is
> verified. The CUDA path (`src/cuda/`, `src/kernels/cuda/cx.cu`) is written
> and now **compiles and links against the real CUDA 13.1 toolkit**
> (nvcc 13.1.115, driverless build via toolkit link stubs — 2026-09-26,
> both build systems). **Run on silicon 2026-09-27:** `PARITY PASSED` +
> `GPU SMOKE PASSED` on the RTX PRO 4000 Blackwell under the hard device
> gate (no soft-gate bypass) after the VRAM tolerance fix (§3.3 item 4).
> The runbook below remains the procedure for (re-)validation; it is
> ordered so any failure is attributable to exactly one layer.

---

## 1. Machine requirements

| Item | Requirement | Notes |
|------|-------------|-------|
| GPU | NVIDIA **RTX PRO 4000 Blackwell** (sm_120, CC 12.0/12.1, 24 GiB, ~672 GB/s) | Target profile `MM_TARGET_PRO4000` (`src/config/config.c:26`). Enforced as a **hard gate** at engine create (§3.3). Any other NVIDIA GPU works for validation via the soft-gate env var (`MIMFER_SOFT_DEVICE_GATE`, §3.2). |
| OS | Linux x86-64 (POSIX) | The tree uses POSIX headers (`unistd.h`, `sys/mman.h`, `fcntl.h` — in `src/artifact/artifact.c`, which is not part of the validation builds, but keep the tree on Linux). |
| CUDA toolkit | **12.8 or newer** (13.x is fine) | sm_120 is supported starting with CUDA 12.8. The *API* floor of the code is 11.0 (launch attributes + L2 access-policy windows); the *device* floor is what forces 12.8+. |
| Driver | Current NVIDIA driver; `nvidia-smi` must list the GPU | |
| Host compiler | GCC (recent; C11) | Builds the golden/self-check directly, and compiles the C translation units inside the nvcc builds. |
| RAM | ≥ 2 GiB free | All tests use a tiny 8-layer built-in model (§3.5); allocations are single-digit MiB. |
| Disk | < 50 MiB | Sources ~1 MB, `/tmp/par_golden.bin` 3.8 MB, four binaries. |

**Build:** the canonical build system is the **root `Makefile`** (GNU Make
only; `make` / `make test` / `make cuda`, artifacts in `build/`). Every
build is still a single `gcc` or `nvcc` command — the Makefile recipes are
the verbatim commands below (identical flags), so the manual form remains
the reference. **There are no external dependencies:** libc + libm +
`cuda_runtime.h` only.

## 2. What gets built and run

| # | Binary | Builder | Purpose | Pass line |
|---|--------|---------|---------|-----------|
| 1 | `/tmp/par_golden` | gcc (host) | Runs the CPU reference op-by-op and writes the golden file `/tmp/par_golden.bin` | `PAR GOLDEN WRITTEN` |
| 2 | `/tmp/par_selfcheck` | gcc (host) | Replays the golden against this machine's CPU reference — the **oracle pre-gate**: proves the golden matches this machine bit-exactly before any GPU comparison | `PARITY PASSED` |
| 3 | `/tmp/parity_test` | nvcc | Per-op numeric parity: GPU op outputs vs the golden, all 18 executable opcodes, three tolerance classes | `PARITY PASSED` |
| 4 | `/tmp/gpu_smoke` | nvcc | End-to-end engine: graph mode, direct mode, determinism, CPU-oracle token heads, clean final device sync | `GPU SMOKE PASSED` |

**Order matters: 1 → 2 (gate) → 3 → 4.** If step 2 fails on the GPU
machine, stop: a later parity failure could not be attributed to the GPU
(rebuild the golden there instead — §5.3).

## 3. Environment

### 3.1 Working directory
All commands assume **CWD = repository root** (relative `-I` and source
paths). Binaries and the golden land in `/tmp` (paths in the commands are
all substitutable).

### 3.2 Environment variables
- **`MIMFER_SOFT_DEVICE_GATE`** (any value, e.g. `=1`) — downgrades the hard
  device-profile gate (§3.3) to a warning. **Required on any GPU that is not
  the target card** (test rig); not needed on a real RTX PRO 4000 Blackwell,
  which passes the hard gate natively (verified on silicon 2026-09-27 after
  the VRAM tolerance fix, §3.3 item 4). (`src/engine/engine.c:361`)
- Nothing else is read from the environment.

### 3.3 Device gate (checked at `mm_engine_create`, MM_WITH_CUDA builds)
Hard checks (fail → `MM_ERR_DEVICE`, engine is not created):
1. at least one CUDA device visible (`cudaGetDeviceCount`);
2. device name contains `"RTX PRO 4000"`;
3. compute capability 12.x (12.0 or 12.1);
4. total VRAM within 5% below 24 GiB (`MM_VRAM_TOLERANCE_PCT`): the
   driver-reported total sits below the datasheet nominal — the
   driver/firmware reserves part of the card (the 24 GB card reports
   25,153,044,480 bytes = 23.4256 GiB, measured 2026-09-27, driver
   595.71.05);
5. derived bandwidth (driver `memoryClockRate × memoryBusWidth × 2`) within
   ±10 % of 672 GB/s.

Soft checks (logged at startup, **never fatal**): L2 cache size, shared
memory per SM. The startup log prints the probed values — record them in
your report.

### 3.4 Build flags — and why (do not change without re-running the parity gate)
| Flag | Why |
|------|-----|
| `-fmad=false` (nvcc) | The CPU oracle is compiled **without FMA contraction**. Allowing FMA on the GPU changes f32 accumulation rounding and would break the 1-bf16-ulp parity gate on computed ops. |
| `-DMM_WITH_CUDA` | Enables the real CUDA paths in `cuda_rt.c`, `cuda_mem.c`, `alloc.c`, `engine.c`, `config.c`. Without it the tree builds the host reference (that is steps 1–2). |
| `-arch=sm_120` | Must match the GPU. The commands below target the PRO 4000 Blackwell. **On a different GPU (soft-gated), replace with that GPU's arch** (e.g. `-arch=sm_90`, `-arch=sm_80`). Without `-arch`, nvcc uses the toolkit default, which may not produce code that runs on your device. |
| language **by file extension** (no `-x` anywhere) | nvcc compiles the `.c` sources as **C** and `cx.cu` as CUDA from the extension alone. The dispatcher ABI is pinned to C by the `extern "C"` guards in the public headers (`include/mimfer/*.h`, `src/model/tensor_registry.h`). **Do NOT pass `-x` per source**: in CUDA 13.1 it is a *global* last-value-wins option (nvcc warns "incompatible redefinition for option 'x'"), so any `-x` on the line forces one language on **all** files — the old per-source `-x c` recipes compiled `cx.cu` as plain C (no `__global__`, no `<<<>>>`). Verified against the real toolkit, 2026-09-26. |
| `-O2` (nvcc) | Used by the verified toolchain; keep it for comparability. |
| host builds | `-std=c11 -Wall -Wextra -Werror -lm` |
| include paths | `-Iinclude -Isrc/model -Isrc/kernels` — `src/model` (tensor_registry.h) and `src/kernels` (kx.h) are needed by the engine-linked builds; the step-1 plan-only build needs only `-Iinclude`. |

Note: `cx.cu` uses **no architecture-specific intrinsics** (bf16 is done in
software via the accessors in `include/mimfer/tensor.h`; no `__CUDA_ARCH__`
guards, no FMA) — it compiles for any `-arch`; the `-arch` choice is driven
by the *device*, not the code.

### 3.5 Test model
The engine runs a built-in **tiny hybrid shape** (`host_tiny_shape`,
`src/engine/engine.c`): `[lin,lin,lin,full] × 2` → 8 layers
(`n_full=2`, `n_lin=6`), standing in for artifact parsing (the artifact
loader is not wired in this build — `engine.c:377`). The production shape
constants (Qwen3.8-27B, `config.h`) are used for sizing only. Device memory
usage is negligible.

**Sampling is greedy everywhere** (`temperature = 0` / `top_k = 0`). CUDA
graph capture is greedy-only by design (the non-greedy SAMPLE launcher
stages logits through host memory, which a graph cannot express) — which is
also why every validation run is greedy.

## 4. Step 0 — Sanity (5 minutes)

```sh
nvidia-smi                                   # GPU visible? note model + driver
nvcc --version                               # CUDA >= 12.8
gcc --version
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv
```

Decide now whether `MIMFER_SOFT_DEVICE_GATE=1` is needed (anything that is
not a "RTX PRO 4000" card). If the card is not sm_120, fix the `-arch` in
§6/§7 commands **before** starting.

## 5. Step 1 — Golden + host oracle pre-gate (gcc; no GPU needed)

### 5.1 Build and run the golden writer
```sh
gcc -std=c11 -Wall -Wextra -Werror -Iinclude -Isrc/model -Isrc/kernels \
    tests/host/par_golden.c src/engine/engine.c src/plan/plan.c \
    src/alloc/alloc.c src/config/config.c src/core/mimfer.c \
    src/cuda/cuda_rt.c src/cuda/cuda_mem.c src/sampling/sampling.c \
    src/kv/kv.c src/kernels/cpu/cx.c src/kernels/kx.c \
    src/model/tensor_registry.c src/rope/rope.c src/sched/sched.c \
    -lm -o /tmp/par_golden
/tmp/par_golden /tmp/par_golden.bin
```
Expected (both lines):
```
  coverage: CPU reference covers 18/18 executable opcodes
PAR GOLDEN WRITTEN /tmp/par_golden.bin (1 prefill + 16 decode rounds, seed 12345)
```
The golden is 3,761,596 bytes: per-op output records
(`opcode, layer, n, ulp_class, offset`, 20 bytes) + output bytes, plus a
header (`MAGIC='MMPAR001'`, seed, prefill/decode token counts). Format spec:
header of `tests/host/par_golden.c`.

### 5.2 Build and run the self-check (oracle gate)
```sh
gcc -std=c11 -Wall -Wextra -Werror -Iinclude -Isrc/model -Isrc/kernels \
    tests/cuda/parity_test.c src/engine/engine.c src/plan/plan.c \
    src/alloc/alloc.c src/config/config.c src/core/mimfer.c \
    src/cuda/cuda_rt.c src/cuda/cuda_mem.c src/sampling/sampling.c \
    src/kv/kv.c src/kernels/kx.c src/kernels/cpu/cx.c \
    src/model/tensor_registry.c src/rope/rope.c src/sched/sched.c \
    -lm -o /tmp/par_selfcheck
/tmp/par_selfcheck /tmp/par_golden.bin
```
Expected (all five lines; the self-check is CPU-vs-CPU on the same machine,
so it passes **bit-exact** — a stronger result than the tolerance gate):
```
  coverage: CPU reference covers 18/18 executable opcodes
  compared 1564 ops, 2108 observables, 2009424 elements (2009424 bit-exact)
  max deviation class 1 (computed bf16): 0 f32 ulp = 0.000 bf16 ulp (tol 1 bf16 ulp)
  max deviation class 2 (f32 state / bf16 conv tail): 0 f32 ulp (tol 4 f32 / 1 bf16 ulp)
PARITY PASSED (/tmp/par_golden.bin: 1 prefill + 16 decode rounds)
```
**Gate rule:** `par_selfcheck` must print `PARITY PASSED` with
`2009424 bit-exact` before any GPU comparison is meaningful. It proves this
machine's CPU reference reproduces the golden through the exact walk the
GPU test uses (reader, observable sizing, control-buffer fills, token
chaining, comparator).

### 5.3 Golden provenance rule
The golden in `/tmp` here (the dev host) was written by the dev host's CPU.
If 5.2 **fails on the GPU machine**, the golden and this machine's CPU
reference disagree (different compiler/march/FP behavior) — do **not**
touch GPU code: rebuild the golden on the GPU machine (5.1) and re-run 5.2.
The GPU machine's own CPU remains the oracle for that session.

## 6. Step 2 — Parity test (nvcc; per-op numeric gate)

### 6.1 Build (GPU machine; `MIMFER_SOFT_DEVICE_GATE=1` if not the target card)
```sh
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
```
No `-x` flags: nvcc compiles `.c` as C and `cx.cu` as CUDA by extension
(§3.4). On a build host **without the GPU driver**, append the toolkit link
stub to the command: `-L<cuda toolkit>/lib64/stubs` (Makefile:
`make cuda NVCC_EXTRA=-L<toolkit>/lib64/stubs`; CMake:
`-DMIMFER_ENABLE_CUDA=ON -DMIMFER_NVCC_EXTRA="-L<toolkit>/lib64/stubs"`).
### 6.2 Run
```sh
/tmp/parity_test /tmp/par_golden.bin
```
### 6.3 Expected output
```
  coverage: GPU launchers cover 18/18 executable opcodes
  compared 1564 ops, 2108 observables, 2009424 elements (N bit-exact)
  max deviation class 1 (computed bf16): 0 f32 ulp = 0.000 bf16 ulp (tol 1 bf16 ulp)
  max deviation class 2 (f32 state / bf16 conv tail): 0 f32 ulp (tol 4 f32 / 1 bf16 ulp)
PARITY PASSED (/tmp/par_golden.bin: 1 prefill + 16 decode rounds)
```
where `N` (the `bit-exact` count) is typically below the total: class 1/2
elements may deviate by ≤1 bf16 ulp / ≤4 f32 ulp and still pass. The
coverage line verifies the GPU launcher set mirrors the CPU reference set
(`kx_cuda_oppref` == `kx_cpu_oppref` for every executable opcode).
Notes:
- class 0 ops (EMBED, COPY, SAMPLE, KVSTORE data paths): enforced
  **bit-exact element-wise** — any differing byte is an immediate FAIL, so
  a pass means there is no `bit-exact mismatch` line;
- class 1 ops (all bf16-computed: norms, GEMMs, attention, conv,
  DeltaNet): ≤ **1 bf16 ulp** per element (both GPU and golden quantize
  f32→bf16 at the end);
- class 2 (fp32 recurrence/conv state, read back after the round):
  ≤ **4 f32 ulp** per element (1 bf16 ulp where the stored state is bf16,
  the conv tail);
- exit code 0 on pass; on failure the binary prints the **first failing
  element verbatim** (e.g. `FAIL decode r5 op41 tag2: element 1024/5120
  deviates 2 f32 ulp (tol 1): golden ... got ...`, or
  `FAIL prefill op3 tag1: bit-exact mismatch at byte 7/128 (golden 3f, got 40)`),
  then the per-class maxima, and returns 1. Report that first-FAIL line
  (§10).

## 7. Step 3 — GPU smoke test (nvcc; end-to-end plumbing gate)

### 7.1 Build
```sh
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
```
### 7.2 Run
```sh
/tmp/gpu_smoke
```
### 7.3 Expected output
```
  short (block 0)        mode=graph+direct rounds=17 tokens_out=17 head=[451 20 430 317 0 152 24 414]
  long (crosses block)   mode=graph+direct rounds=71 tokens_out=71 head=[127 230 247 387 485 327 294 387]
GPU SMOKE PASSED
```
The token heads must equal the **CPU oracle heads** (the host
`engine_smoke` outputs for the same cases/seed).

### 7.4 What the binary verifies internally
Per case (`short`: 4-token prompt + 16 decode, seed 12345; `long`: 4 + 70
decode crossing the 64-token KV block boundary, seed 99) it runs **four
full engine lifecycles** — graph A, graph B, direct A, direct B — and
checks:
1. every `mm_status` is `MM_OK` (device gate + probe, arenas, pools, plan
   build, graph capture where applicable, every round step);
2. round count = 1 prefill + N decode in all four runs;
3. the token counter matches across all runs;
4. graph A == graph B and direct A == direct B (**determinism within mode**);
5. graph stream == direct stream (**cross-mode agreement**);
6. first 8 tokens match the CPU oracle head;
7. a final `mm_device_sync_all()` is clean (catches async errors the per-op
   syncs could have missed).

### 7.5 Independent determinism confirmation
Run the binary a second time and diff the full output; the two runs must be
byte-identical (seeds are fixed in the test):
```sh
/tmp/gpu_smoke > /tmp/smoke1.txt 2>&1
/tmp/gpu_smoke > /tmp/smoke2.txt 2>&1
diff /tmp/smoke1.txt /tmp/smoke2.txt && echo DETERMINISTIC
```

## 8. Step 4 (optional) — Sanitizer sweep

Deeper memory/parity sweep of the smoke binary (slower):
```sh
compute-sanitizer --tool memcheck /tmp/gpu_smoke
```
(or the equivalent `compute-sanitizer --tool racecheck` / `initcheck`).
Expected: the usual `GPU SMOKE PASSED` lines and an error summary of **0**.
Any finding must be reported per §10 (first finding verbatim).

## 9. Troubleshooting

### 9.1 Build failures (nvcc step)
| Symptom | Cause | Action |
|---------|-------|--------|
| `error: unsupported gpu architecture 'compute_120'` | Toolkit < 12.8 | Upgrade to CUDA 12.8+, or (on a non-target card) use that card's `-arch` + `MIMFER_SOFT_DEVICE_GATE=1`. |
| `cuda_runtime.h: No such file or directory` | Toolkit not installed / not on PATH | Install the toolkit; verify `nvcc --version`. |
| `undefined reference to kx_op_embed` (or other `kx_op_*` / `kx_cuda_oppref`) | `-DMM_WITH_CUDA` missing, or `src/kernels/cuda/cx.cu` missing from the command | Use the verbatim command from §6.1/§7.1. |
| `undefined reference to _Z16kx_cuda_opprefP...` (mangled symbol) | the `extern "C"` guards in the public headers (`include/mimfer/*.h`, `src/model/tensor_registry.h`) were altered | Restore the guards (do not "clean up" the `#ifdef __cplusplus` blocks). |
| C-compile errors inside `.c` files under nvcc | host-compiler drift (nvcc compiles the C TUs with the system C compiler) | Build the identical source list with plain `gcc` (step-1 style) to split host-TU errors from `.cu` errors; report which layer fails. |
| `undefined reference to cuInit` / `cuDeviceGet*` (driver API) | host has the toolkit but **no GPU driver** — the driver API (`-lcuda`) cannot resolve | Link the toolkit's driver stub: append `-L<cuda toolkit>/lib64/stubs` to the command (Makefile: `NVCC_EXTRA`, CMake: `-DMIMFER_NVCC_EXTRA`). Link-only; no codegen impact. Run the binaries on the GPU machine. |
| `cx.cu` compile errors | toolkit-version drift or a tree edit (cx.cu is known-good on nvcc 13.1.115 as of 2026-09-26) | Report verbatim (nvcc version, full message, line number); first check `nvcc --version` matches 12.8+/13.x expectations, then treat it as a regression against the verified build. |

### 9.2 Runtime failures (at engine create / load)
| Symptom | Cause | Action |
|---------|-------|--------|
| `device gate: name "..." does not contain "RTX PRO 4000"` / `compute capability ... need 12.x` / `device gate: N GiB < required 24 GiB (5% tolerance, floor 22 GiB)` / `derived bandwidth ... outside [...]` | non-target card (test rig) or wrong device selected | `export MIMFER_SOFT_DEVICE_GATE=1` (soft gate) and/or check `CUDA_VISIBLE_DEVICES`. On the real PRO 4000, a bandwidth gate failure means the driver-reported clocks/bus differ from the ESTIMATE in `config.c` — report the logged probed values. (The VRAM check already tolerates the driver reserve — 23.4256 GiB reported on the 24 GB card — so a VRAM failure on the real card means a genuinely different card or a driver regression.) |
| `engine: device profile gate failed (...); continuing (MIMFER_SOFT_DEVICE_GATE set)` | soft gate active | Informational; expected on non-target silicon. |
| `cuda: <api> (line N): invalid device function` | `-arch` doesn't match the GPU | Rebuild with the GPU's `-arch` (§3.4). |
| `cuda: node window rejected on node N (continuing without it)` | L2 access-policy window not applicable on this driver/device | **Expected soft warning, not a failure** (the window is an optimization; commit continues without it). Record it in the report. |
| Graph capture error at load (`StreamEndCapture` fails, `operation not supported when stream is capturing`, …) | capture-mode illegality in the op sequence | Real engine bug: report the line, the plan being captured (prefill / decode M, greedy), and the full log. Do not work around it. |
| `cudaMalloc(...) failed` (OOM) | VRAM occupied by another process | `nvidia-smi` — the test uses single-digit MiB; anything else holding the GPU is the cause. |

### 9.3 Parity failures (`/tmp/parity_test`)
1. **Check the pre-gate first:** if `par_selfcheck` does not pass on this
   machine, rebuild the golden here (§5.3) — do not debug GPU code.
2. Read the **first FAIL line** — `FAIL <prefill | decode rN> op# tagD: …`,
   where the detail is `bit-exact mismatch at byte I/N (golden XX, got YY)`
   (class 0) or `element I/N deviates D f32 ulp (tol T): golden G got F`
   (classes 1/2); the per-class maxima are printed at the end.
3. Interpret by class:
   - **class 0 (bit-exact) fail** (EMBED/COPY/SAMPLE/KVSTORE data paths):
     not a rounding issue — data-path bug (copy size, offset, pool
     indexing). Report op + layer + offset verbatim.
   - **class 1 at exactly 1 ulp** on computed ops while classes 0 and 2
     pass: most likely FMA contraction. Verify the binary was built with
     `-fmad=false` and the golden by a non-FMA host compiler (the
     self-check gate proves the oracle side is clean; if so, the failure is
     genuine and must be reported).
   - **class 2 (f32 state) fail**: recurrence/conv state disagreement —
     report op + layer; compare the lin/conv state handling against the
     launcher comments in `cx.cu`.
4. Never "fix" a parity failure by widening the tolerances in
   `tests/host/par_golden.c` — the tolerances are the gate definition.

### 9.4 Smoke failures (`/tmp/gpu_smoke`)
- Any `FAIL tests/cuda/gpu_smoke.c:LINE: <msg>` line identifies the check
  (the §7.4 list). Most informative: `graph and direct dispatch agree`
  (plumbing difference between the graph and per-op paths) and
  `token stream matches the CPU oracle` (numeric divergence beyond the
  per-op gate — cross-reference §9.3).
- Non-determinism between two runs: capture both outputs, re-run under
  `compute-sanitizer --tool racecheck`, report the first finding.

### 9.5 Sanitizer findings
Report the **first finding verbatim** (error class, kernel name, line,
address/size), the step that triggered it (§6/§7/§8), and the full summary
line. Do not re-run with different flags to make it disappear.

## 10. Bug report template

```
mimfer GPU validation report
Date / machine:
  GPU:                 (nvidia-smi: name, CC, VRAM, driver)
  CUDA toolkit:        (nvcc --version, verbatim)
  GCC:                 (gcc --version)
  MIMFER_SOFT_DEVICE_GATE: set / not set
  Probed device line from startup log:
Step (0/1/2/3/4):  <sanity | golden | selfcheck | parity | smoke | sanitizer>
Result:            PASS / FAIL
First failing line (verbatim):
  e.g. "FAIL decode r5 op41 tag2: element 1024/5120 deviates 2 f32 ulp (tol 1): golden 0.75 got 0.75390625"
      or "cuda: cudaGraphLaunch (line 258): invalid device function"
Full output of the step (attach file if long):
Per-class maxima (parity):
Sanitizer summary line (if applicable):
```

## 11. Hard invariants (do not "fix")

- `-fmad=false` on all GPU builds; the CPU oracle stays FMA-free.
- `extern "C"` guards in `include/mimfer/kernels.h` and `src/kernels/kx.h`
  (C/C++ linkage of the dispatcher ABI).
- The golden is written **only** by the CPU reference (`par_golden`); it is
  never regenerated from GPU output.
- Tolerance classes as defined in `tests/host/par_golden.c`: 0 = bit-exact,
  1 = ≤1 bf16 ulp, 2 = ≤4 f32 ulp.
- Graph capture is greedy-only; the non-greedy SAMPLE path stays
  graph-incapable by design.
- The device gate is a hard check unless `MIMFER_SOFT_DEVICE_GATE` is set.
- Any change to the CPU reference, the golden writer, or the tolerances
  requires re-running the host regression (`plan_test`, `engine_smoke`,
  `par_golden`, `par_selfcheck`) and re-stating the expected outputs.
