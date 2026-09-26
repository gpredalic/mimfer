# BUILD_AUDIT.md — mimfer final build audit (release candidate)

> **Date:** 2026-09-26
> **Scope:** the entire source tree (`include/`, `src/`, `tests/`, root
> build system) audited as release-candidate code before RTX PRO 4000
> Blackwell validation. Method: full file inventory, per-translation-unit
> include-resolution check, build-registration cross-check (every source
> file mapped to the target(s) that compile it), empirical host build
> (`make clean && make && make test`), `-fsyntax-only` compilation of
> unregistered TUs under the exact release flags, Makefile
> target/`.PHONY`/prerequisite audit, dry-run expansion of the GPU
> recipes (`make -n`), and cross-check of the verbatim `gcc`/`nvcc`
> commands in `GPU_VALIDATION.md` §5–§7 against the Makefile recipes.
>
> **Verdict:** the host build system is sound and hermetic (see B5); two
> **critical GPU-build defects** were found and fixed (B1: `cx.cu` missing
> from the nvcc link line; B10: no provider of `kx_cpu_oppref` in the GPU
> link set); the remaining findings are documentation/hygiene items.
> Nothing in this audit added a feature, an optimization, or an
> architecture change — the only code changes are the GPU link fixes in
> B1 (Makefile) and B10 (one guarded reference copy in `parity_test.c`).

---

## 1. Findings

Severity: **critical** = blocks the planned GPU validation · **fixed** =
resolved in this session · **accepted** = intentional, documented ·
**info** = no action taken.

| # | Sev | Finding | Status |
|---|-----|---------|--------|
| B1 | critical | Makefile GPU recipes dropped `src/kernels/cuda/cx.cu` from the nvcc link line | **fixed this session** |
| B10 | critical | GPU `parity_test` link set had no provider of `kx_cpu_oppref` (would fail to link even after B1) | **fixed this session** (guarded reference copy in `parity_test.c`) |
| B2 | accepted | 3 TUs registered in no build target (`artifact.c`, `tokenizer.c`, `telemetry.c`) | documented (pre-existing, §3 of RELEASE_READINESS.md) |
| B3 | accepted | `flags.c` registered only in the `flags_test` build | documented (no CLI binary exists yet) |
| B4 | info | "CMake vs Make parity" check is **not applicable** — there is no CMake in the tree | documented below |
| B5 | info | Host build is hermetic: no CUDA toolkit required (positive finding) | verified |
| B6 | info | `HDRS` hard-codes the two in-tree private headers | noted (caveat only) |
| B7 | info | 3 untracked `tmux-*.log` junk files at repo root (≈1.3 GB total) | deletion = human decision; sizes in R6 corrected |
| B8 | info | `cx.cu` has never been compiled (no nvcc on this host) | pre-existing top risk (R1), unchanged |
| B9 | doc | README Extensibility section carried a stale "no Makefile to keep in sync" sentence | **fixed this session** |

### B1 — GPU recipes omitted `cx.cu` from the nvcc command line (fixed)

**Evidence.** The GPU recipes compiled `$(GPU_C_SRCS)` — the 13 `.c` files
with a `-x c` prefix — but never the `.cu` file that is in `$(GPU_SRCS)`
as a prerequisite:

```
nvcc -O2 -fmad=false -DMM_WITH_CUDA -arch=sm_120 -Iinclude -Isrc/model -Isrc/kernels \
     -x c tests/cuda/parity_test.c -x c src/engine/engine.c ... -x c src/sched/sched.c \
     -o build/parity_test            # ← no src/kernels/cuda/cx.cu
```

(concrete pre-fix expansion, captured with `make -n build/parity_test`).

**Why it is fatal on a GPU machine.** In the GPU build,
`src/kernels/cpu/cx.c` is filtered out of the source set and
`src/kernels/cuda/cx.cu` replaces it: it defines all 18 `kx_op_*`
launcher implementations that `mm_kx_invoke()` in `src/kernels/kx.c`
dispatches to (verified: `mm_status kx_op_*` defined 18× in `cpu/cx.c`
and 18× in `cuda/cx.cu`), plus `kx_cuda_oppref()`, which
`tests/cuda/parity_test.c:642,644` calls directly (prototype in
`src/kernels/kx.h:43`, "DEFINED in the C++ translation unit cx.cu").
Without `cx.cu` on the link line, `make cuda` fails with undefined
references to `kx_op_*` / `kx_cuda_oppref` — i.e. the first step of the
RTX PRO 4000 validation would have been a guaranteed link failure.

**Why it was not caught earlier.** This host has no `nvcc`, so the
recipes never executed; `make cuda` stops at `check-nvcc` first. The
*documented* verbatim commands in `GPU_VALIDATION.md` §6.1/§7.1 **do**
include `cx.cu`, so the runbook was correct and the Makefile had drifted.

**Fix.** One variable + one token per recipe:

```make
GPU_CU_SRCS := $(filter %.cu,$(GPU_SRCS))
...
$(NVCC) $(NVCCFLAGS) -x c tests/cuda/parity_test.c $(GPU_C_SRCS) $(GPU_CU_SRCS) -o $@
$(NVCC) $(NVCCFLAGS) -x c tests/cuda/gpu_smoke.c   $(GPU_C_SRCS) $(GPU_CU_SRCS) -o $@
```

`cx.cu` is passed **without** `-x c` (it must compile as CUDA/C++),
matching the documented commands. Post-fix `make -n` expansion shows
`... -x c src/sched/sched.c src/kernels/cuda/cx.cu -o build/parity_test`.

**Residual.** The nvcc link itself is still unexecuted here (no nvcc).
The first `make cuda` on the GPU machine is the verification; the
runbook's §9.1 build-failure table already covers the isolated `cx.cu`
compile/link case (R1).

### B10 — GPU `parity_test` link set had no provider of `kx_cpu_oppref` (fixed)

Found while verifying the B1 fix. `tests/cuda/parity_test.c`
`check_coverage()` (called unconditionally from `main`) asserts the GPU
launcher coverage mirrors the CPU reference:

```c
#ifdef MM_WITH_CUDA
    CHECK(kx_cuda_oppref(op) == kx_cpu_oppref(op), ...);
#else
    CHECK(kx_cpu_oppref(op) == 1, ...);
#endif
```

`kx_cpu_oppref` is defined **only** in `src/kernels/cpu/cx.c:668`;
`cx.cu` defines only `kx_cuda_oppref` (line 1018). The GPU source set
(`GPU_SRCS`) filters out `cpu/cx.c`, and it must not be added back: it
defines the same 18 `kx_op_*` launchers as `cx.cu`, so linking both TUs
would multiply-define them. Result: even with B1 fixed, the GPU
`build/parity_test` (and the documented §6.1 verbatim command, which
omits `cpu/cx.c` for the same reason) still failed to link with
`undefined reference to kx_cpu_oppref`.

**Fix.** A reference copy of the 18-opcode coverage vector is now
defined in `tests/cuda/parity_test.c` under `#ifdef MM_WITH_CUDA`
(placed immediately above `check_coverage()`). The invariant is:

- **GPU build:** the test's copy is compiled (define on); `cpu/cx.c` is
  not linked → exactly one definition, link succeeds; the mirror
  assertion compares `cx.cu`'s table against the reference vector.
- **Host self-check:** the define is off → the copy is not compiled; the
  real definition comes from `cpu/cx.c` (in `ENGINE_SRCS`) → the host
  recipe is **byte-identical to before** (golden bit-determinism
  untouched), and `make test` remains green (re-verified).
- The copy must stay identical to `kx_cpu_oppref()` in `cpu/cx.c` (same
  18 executable opcodes, `default: 0`) — stated in the code comment.
  Note the assertion still verifies what the design intends at the GPU
  binary level (CUDA launcher set vs. the CPU-reference coverage
  vector); cross-file drift between `cpu/cx.c` and the test vector is
  caught by the golden/parity data comparison and code review, not by
  this symbol check.

**Verification on this host.** The host suite stays green (the guarded
block is inactive there); the exact guarded block was extracted and
compiled standalone with `-std=c11 -Wall -Wextra -Werror
-Iinclude -Isrc/model -Isrc/kernels` and the opcode enums from
`mimfer/plan.h` — clean. A full `-DMM_WITH_CUDA` compile of the whole TU
needs `<cuda_runtime.h>` (absent here), so like B1 the nvcc link itself
remains verified on the GPU machine.

### B2 — TUs registered in no build target (accepted, pre-existing)

`src/artifact/artifact.c` (448 lines), `src/tokenizer/tokenizer.c`
(285), `src/telemetry/telemetry.c` (118) are written but intentionally
unintegrated (their headers `artifact.h`/`tokenizer.h`/`telemetry.h` are
already included by `mimfer/engine.h`). Audit check performed this
session: all three pass
`gcc -std=c11 -Wall -Wextra -Werror -Iinclude -Isrc/model -Isrc/kernels
-fsyntax-only` cleanly, so they are **not** latent build breakage — but
they are compiled by no Makefile target, so any drift is only detected at
integration time. Consistent with `RELEASE_READINESS.md` §3/§7;
integration is future work, out of scope for this audit.

### B3 — `flags.c` registered only in its unit-test build (accepted)

`src/flags/flags.c` appears only in `FLAGS_TEST_SRCS` (the
`build/flags_test` binary). There is no CLI/server binary in the tree;
the flag-grammar module is consumed by a future CLI, so "built and
exercised only by `flags_test`" is the correct registration today. Not
dead code: `tests/host/flags_test.c` covers the full grammar and
cross-field validation and runs in `make test`.

### B4 — CMake/Make parity: not applicable (Make is the sole build system)

The mission asked to "check that CMake and Make build the same
targets". **There is no CMake build system in the tree**: no
`CMakeLists.txt` (or any CMake file) anywhere; the Makefile header says
"By design there is no CMake/Meson/Bazel or any other framework"; the
README says "GNU Make only — no CMake/Meson/Bazel". The parity check is
therefore vacuous; the single source of build truth is the root
`Makefile`.

Residual CMake traces (all harmless, none touched):

- `.gitignore` carries template CMake/Ninja ignore patterns
  (`CMakeFiles/`, `CMakeCache.txt`, `cmake_install.cmake`,
  `compile_commands.json`, `.ninja*`);
- the branch name `feature/cmake-flag-surface` is a historical artifact —
  the feature is the CLI flag *surface*, not a CMake build.

**Make ↔ documentation parity (the real check).** The verbatim commands
in `GPU_VALIDATION.md` were cross-checked against the Makefile recipes:

| Documented command | Makefile target | Flags identical? | Source set identical? |
|--------------------|-----------------|------------------|------------------------|
| §5.1 golden (`gcc`) | `build/par_golden` | yes (`-std=c11 -Wall -Wextra -Werror -Iinclude -Isrc/model -Isrc/kernels -lm`) | yes (test + `ENGINE_SRCS`) |
| §5.2 self-check (`gcc`) | `build/par_selfcheck` | yes | equivalent — docs compile `tests/cuda/parity_test.c` directly; the Makefile compiles the `tests/host/par_selfcheck.c` wrapper that `#include`s that same file (one source of truth) |
| §6.1 `parity_test` (`nvcc`) | `build/parity_test` | yes (`-O2 -fmad=false -DMM_WITH_CUDA -arch=sm_120` + same `-I` set) | yes (test + 13× `-x c` `.c` + `cx.cu`; docs place `cx.cu` mid-list, order is irrelevant) |
| §7.1 `gpu_smoke` (`nvcc`) | `build/gpu_smoke` | yes | yes (same shape) |

Before the B1 fix, the Makefile GPU recipes were the *only* place `cx.cu`
was dropped; after the fix, Make and documentation agree on every target.

### B5 — Host build is hermetic (positive finding)

`cuda_runtime.h` does **not** exist on this host (verified: `find /usr
/opt`, and a bare `cc` probe fails), yet `make` and `make test` build and
pass. Every `<cuda_runtime.h>` include is guarded:
`include/mimfer/cuda_rt.h` keeps its public surface CUDA-free (the
`#include <cuda_runtime.h>` sits under `#ifdef MM_WITH_CUDA`, which the
host build never defines), and `cuda_rt.c` / `cuda_mem.c` / `config.c` /
`alloc.c` guard their direct includes with `#if defined(MM_WITH_CUDA)`
(host branch = honest no-op stubs). Only `cx.cu` and the `tests/cuda`
TUs see the real CUDA runtime, and only under nvcc.
**Consequence: the host build needs only gcc (C11) + libm — no CUDA
toolkit, no GPU, no driver.** The `MM_WITH_CUDA` define is set by the
Makefile exactly on the nvcc recipes (`NVCCFLAGS`), never on host
recipes.

### B6 — `HDRS` hard-codes the two private headers (caveat)

`HDRS := $(wildcard include/mimfer/*.h) src/kernels/kx.h
src/model/tensor_registry.h` covers every header that exists today
(verified complete). If a future module adds a private header in a new
directory, it will silently be missing from the header-change
prerequisite set: builds would still succeed, but edits to that header
would no longer force a rebuild. Note for the next engineer; no change
made.

### B7 — Repo hygiene: `tmux-*.log` junk (pre-existing, numbers corrected)

`tmux-client-153455.log` (3.5 KB), `tmux-out-153457.log` (11 MB),
`tmux-server-153457.log` (1.3 GB) at the repo root — session junk,
untracked and git-ignored by `*.log`. `RELEASE_READINESS.md` R6 carried
stale sizes (18 KB / 64 MB / 6.4 GB); corrected to the measured 2026-09-26
values. Deletion remains a human decision (R6).

### B8 — `cx.cu` never compiled (pre-existing, unchanged)

No `nvcc` on this host; `cx.cu` (1042 lines) is the only TU never
compiled by any toolchain. It is isolated by the runbook's build order
and the §9.1 failure table. See R1 in `RELEASE_READINESS.md`.

### B9 — Stale README sentence (fixed)

README.md Extensibility said new modules require "registration in the
build (single-line `gcc`/`nvcc` commands — there is no Makefile or CMake
to keep in sync)" — stale from before the root Makefile became canonical
(the same README's Build section correctly names the Makefile as the
canonical build system). Corrected to point at the root `Makefile`.

---

## 2. Build-system conformance matrix

Host toolchain: gcc 13.3.0 (Ubuntu), `cc` = `gcc`; `nvcc`: **absent**
(`make cuda` correctly refuses via `check-nvcc` with an actionable
message — verified).

| Target | Kind | Recipe / sources | Verified on this host |
|--------|------|------------------|------------------------|
| `all` / `host` | build | 7 binaries below | **yes** — clean build, zero warnings under `-Werror` |
| `build/plan_test` | host bin | `plan_test.c` + `PLAN_SRCS` (6) | built + run (`make test`) |
| `build/engine_smoke` | host bin | `engine_smoke.c` + `ENGINE_SRCS` (14) | built + run |
| `build/rope_test` | host bin | `rope_test.c` + `ROPE_TEST_SRCS` (4) | built + run |
| `build/flags_test` | host bin | `flags_test.c` + `FLAGS_TEST_SRCS` (4) | built + run |
| `build/engine_features` | host bin | `engine_features_test.c` + `ENGINE_SRCS` (14) | built + run |
| `build/par_golden` | host bin | `par_golden.c` + `ENGINE_SRCS` (14) | built + run |
| `build/par_selfcheck` | host bin | `par_selfcheck.c` + `ENGINE_SRCS` (14) | built + run |
| `test` / `tests` | test | 6 sequential stages: plan → rope → flags → engine-smoke → engine-features → parity-selfcheck | **green** (2026-09-26, twice: before and after the B1 fix) |
| `golden` / `parity-selfcheck` (+aliases `parity`, `smoke`) | test | golden rewrite + self-check vs golden | **green** — `2009424 elements bit-exact` |
| `cuda` | build (nvcc) | `check-nvcc` + both GPU bins | command lines verified via `make -n` (post-fix B1); execution needs nvcc |
| `gpu-parity` / `gpu-smoke` | build (nvcc) | `build/parity_test`, `build/gpu_smoke` (14 sources each, `cx.cu` included) | same as above |
| `check-nvcc` | gate | nvcc-on-PATH probe | **yes** — refuses cleanly here |
| `clean` / `distclean` / `help` | housekeeping | `rm -rf build/`; no other generated files | **yes** |

`.PHONY` audit: all 18 phony targets are listed; no file target is
marked phony; no phony target is missing. `.DEFAULT_GOAL := all` pins the
default goal (documented Make 4.3 trap). `.DELETE_ON_ERROR:` present.
Header-change rebuilds are covered by `$(HDRS)` prerequisites on every
binary (see B6 caveat).

## 3. Verification evidence (this session, 2026-09-26)

- `make clean && make`: 7 host binaries, zero warnings (`-Wall -Wextra
  -Werror`).
- `make test`: all 6 stages green — plan, rope, flags, engine-smoke,
  engine-features, parity self-check (`1564 ops, 2108 observables,
  2009424 elements (2009424 bit-exact)`).
- `make -n build/parity_test` / `build/gpu_smoke`: pre-fix expansion
  lacked `cx.cu` (B1 evidence); post-fix expansion includes it without
  `-x c`, matching `GPU_VALIDATION.md` §6.1/§7.1.
- `-fsyntax-only` under exact release flags: `artifact.c`,
  `tokenizer.c`, `telemetry.c` all clean (B2).
- B10 probe: the guarded `kx_cpu_oppref` block extracted from
  `tests/cuda/parity_test.c` compiles clean under
  `-std=c11 -Wall -Wextra -Werror` with the project include set (the
  guard active); the host suite re-run after the change is green.
- `make cuda` on this host: clean refusal at `check-nvcc` (actionable
  message).
- `grep -rn "TODO|FIXME|XXX|HACK"` over `src/`, `tests/`, `Makefile`:
  0 hits.
- `git status`: clean before this session's changes; 57 tracked files;
  the working tree is the audited tree (post-audit changes: the B1
  Makefile fix + the documentation syncs listed in §4).
- Hermeticity probe: `cuda_runtime.h` absent from the host; a bare
  `cc` probe including it fails; the Makefile build succeeds (B5).

## 4. Changes made by this audit

1. `Makefile` — B1 fix: `GPU_CU_SRCS` variable; `cx.cu` added to both
   GPU recipes (2 lines + comment).
2. `tests/cuda/parity_test.c` — B10 fix: `#ifdef MM_WITH_CUDA`
   reference copy of `kx_cpu_oppref` above `check_coverage()` (the host
   self-check path is unaffected — the guard is off there).
3. `README.md` — B9 fix (stale Extensibility sentence); Documentation
   table gains this file, `SOURCE_TREE.md`, `MODULE_DEPENDENCIES.md`.
4. `RELEASE_READINESS.md` — R6 sizes corrected (B7); audit section gains
   the B1/B10 findings/fixes.
5. This file, `SOURCE_TREE.md`, `MODULE_DEPENDENCIES.md` — new.
6. `READ_MEMORY.md` — session handoff update (governance rule).

No other source file was modified. No features added, no optimizations,
no architecture change.

## 5. What this host cannot verify (carried to the GPU machine)

- The nvcc compile+link of `build/parity_test` / `build/gpu_smoke`
  (B1 fix is verified at the command-line level only).
- `cx.cu` compilation itself (B8 / R1).
- `par_golden` → `parity_test` parity and `gpu_smoke` (all
  `GPU_VALIDATION.md` §6–§7 gates).

The B1+B10 fixes remove the two guaranteed link failure modes from that
first run; everything else stands as documented in `GPU_VALIDATION.md`
and `VALIDATION_CHECKLIST.md`.

