# VALIDATION_CHECKLIST.md — First GPU Validation Go/No-Go

Run top to bottom on the GPU machine. Commands: full form in
`GPU_VALIDATION.md` (§5–§8); `NVCC_ARCH` = `-arch=sm_120` (or the arch of
the actual card). If any item fails: **stop**, capture evidence per
`GPU_VALIDATION.md` §10, report — do not continue past a failed gate.

**Pass condition: every box checked. First failing item = the finding.**

## Environment (once)
- [ ] `nvidia-smi` shows the GPU — model + driver recorded
- [ ] `nvcc --version` shows CUDA 12.8+ — exact version recorded
- [ ] `gcc --version` recorded
- [ ] CWD = repo root
- [ ] `MIMFER_SOFT_DEVICE_GATE=1` exported **iff** the card is not a
      "RTX PRO 4000" — recorded either way

## Gate 0 — Host oracle pre-gate (no GPU needed)
- [ ] **full host suite green** — `make test` from the repo root prints
      `HOST TEST SUITE PASSED` (plan_test, rope_test, flags_test,
      engine_smoke, engine_features, parity self-check — the feature
      surface is covered before any GPU work)
- [ ] **build succeeds** — `par_golden` + `par_selfcheck` compile clean
      (`-Wall -Wextra -Werror`, zero diagnostics)
- [ ] `/tmp/par_golden /tmp/par_golden.bin` prints
      `PAR GOLDEN WRITTEN ... (1 prefill + 16 decode rounds, seed 12345)`
- [ ] `/tmp/par_selfcheck /tmp/par_golden.bin` prints `PARITY PASSED`
      (with `2009424 bit-exact`)
      (oracle gate: golden matches this machine's CPU bit-exact;
      **fails here ⇒ rebuild golden on this machine, re-run, do not proceed**)

## Gate 1 — Parity (per-op numeric gate)
- [ ] **parity_test builds** (nvcc: `-O2 -fmad=false -DMM_WITH_CUDA
      $NVCC_ARCH`, verbatim command `GPU_VALIDATION.md` §6.1)
- [ ] **parity_test passes** — `/tmp/parity_test /tmp/par_golden.bin` prints:
  - `coverage: GPU launchers cover 18/18 executable opcodes`
  - `PARITY PASSED`
  - no `bit-exact mismatch` FAIL line (class 0 is bit-exact per element)
  - class 1 max deviation `≤ 1 bf16 ulp`; class 2 `≤ 4 f32 ulp`

## Gate 2 — Smoke (end-to-end plumbing gate)
- [ ] **gpu_smoke builds** (nvcc, verbatim command `GPU_VALIDATION.md` §7.1)
- [ ] **gpu_smoke passes** — `/tmp/gpu_smoke` prints `GPU SMOKE PASSED`
- [ ] **graph mode passes** — no `FAIL ... graph run A/B` line; both case
      lines print `mode=graph+direct` (printed only when all four
      lifecycles returned `MM_OK`)
- [ ] **direct mode passes** — no `FAIL ... direct run A/B` line; same
      `mode=graph+direct` lines (direct==graph streams agreed)
- [ ] **deterministic results verified** —
  - in-binary: `deterministic graph-mode token stream` +
    `deterministic direct-mode token stream` checks passed (no FAIL lines),
    and `graph and direct dispatch agree`
  - independent: two runs of `/tmp/gpu_smoke` diff clean (`§7.5`)
- [ ] token heads match the CPU oracle —
  `short: head=[451 20 430 317 0 152 24 414]`,
  `long:  head=[127 230 247 387 485 327 294 387]`
- [ ] round counts as expected: `rounds=17` (short), `rounds=71` (long)

## Gate 3 — Cleanliness
- [ ] **no CUDA runtime errors** — no `cuda: <api> (line N): <err>` log
      lines (the soft warning `node window rejected ... continuing without
      it` is acceptable — record it, do not fail on it)
- [ ] **no sanitizer findings** (optional but recommended) —
      `compute-sanitizer --tool memcheck /tmp/gpu_smoke` ends with 0 errors
      and still prints `GPU SMOKE PASSED`

## Sign-off
- [ ] evidence recorded (outputs verbatim) per `GPU_VALIDATION.md` §10
- [ ] probed device startup line recorded (`device: <name> (sm_12x, ...`)
- [ ] verdict: **PASS** (all gates) / **FAIL** (first failing item + evidence)
