# MODULE_DEPENDENCIES.md — mimfer dependency map

> **Date:** 2026-09-26 (final build audit; companion to `BUILD_AUDIT.md`
> and `SOURCE_TREE.md`). Dependencies are derived from the actual
> `#include` graph of every translation unit (TUs include only project
> headers from `include/mimfer/`, the two in-tree private headers, and
> system headers — no vendored code, no external libraries beyond
> libc/libm on the host and the CUDA runtime under nvcc).

## 1. Per-TU project include graph

System includes omitted. `MM_WITH_CUDA`-guarded includes marked `[cuda]`.

### Public headers (`include/mimfer/`)

| Header | Project includes |
|--------|------------------|
| `mimfer.h` | — (root; system only) |
| `alloc.h` | `mimfer.h` |
| `artifact.h` | `mimfer.h` |
| `config.h` | `mimfer.h` |
| `cuda_mem.h` | `mimfer.h`, `kernels.h` |
| `cuda_rt.h` | `mimfer.h`, `[cuda] <cuda_runtime.h>` |
| `engine.h` | `mimfer.h`, `config.h`, `alloc.h`, `tensor.h`, `kernels.h`, `kv.h`, `plan.h`, `sched.h`, `artifact.h`, `tokenizer.h`, `sampling.h`, `telemetry.h`, `rope.h` |
| `flags.h` | `config.h` |
| `kernels.h` | `mimfer.h`, `plan.h` |
| `kv.h` | `mimfer.h`, `config.h`, `alloc.h` |
| `plan.h` | `mimfer.h`, `config.h`, `cuda_rt.h` |
| `rope.h` | `config.h`, `kernels.h` |
| `sampling.h` | `mimfer.h` |
| `sched.h` | `mimfer.h` |
| `telemetry.h` | `mimfer.h` |
| `tensor.h` | `mimfer.h` |
| `tokenizer.h` | `mimfer.h` |

Header-DAG check: acyclic. `engine.h` is the top hub (12 public
includes); no included header points back to it.

### Private headers

| Header | Project includes |
|--------|------------------|
| `src/kernels/kx.h` | `mimfer/kernels.h`, `mimfer/plan.h` |
| `src/model/tensor_registry.h` | `mimfer/engine.h` |

### Library TUs

| TU | Project includes |
|----|------------------|
| `src/core/mimfer.c` | `mimfer.h`, `tensor.h` |
| `src/config/config.c` | `cuda_rt.h`, `sampling.h`, `config.h`, `[cuda] <cuda_runtime.h>` |
| `src/alloc/alloc.c` | `alloc.h`, `[cuda] <cuda_runtime.h>` |
| `src/sampling/sampling.c` | `sampling.h` |
| `src/cuda/cuda_rt.c` | `cuda_rt.h`, `[cuda] <cuda_runtime.h>` |
| `src/cuda/cuda_mem.c` | `cuda_mem.h`, `[cuda] <cuda_runtime.h>` |
| `src/plan/plan.c` | `plan.h`, `engine.h`, `kernels.h` |
| `src/kv/kv.c` | `cuda_rt.h`, `kv.h` |
| `src/sched/sched.c` | `sched.h` |
| `src/rope/rope.c` | `rope.h` |
| `src/flags/flags.c` | `flags.h` |
| `src/engine/engine.c` | `engine.h`, `cuda_mem.h`, `tensor_registry.h` (private) |
| `src/kernels/kx.c` | `kernels.h`, `engine.h` |
| `src/kernels/cpu/cx.c` | `kx.h` (private), `engine.h`, `tensor.h` |
| `src/kernels/cuda/cx.cu` | `kernels.h`, `engine.h`, `tensor.h`, `cuda_rt.h`, `kx.h` (private), `<cuda_runtime.h>` (unguarded — nvcc-only TU) |
| `src/model/tensor_registry.c` | `engine.h`, `tensor.h`, `tensor_registry.h` (private) |
| `src/artifact/artifact.c` | `artifact.h` (unregistered in builds — B2) |
| `src/tokenizer/tokenizer.c` | `tokenizer.h` (unregistered — B2) |
| `src/telemetry/telemetry.c` | `telemetry.h`, `mimfer.h` (unregistered — B2) |

### Test TUs

| TU | Project includes |
|----|------------------|
| `tests/host/plan_test.c` | `engine.h`, `kernels.h`, `plan.h` |
| `tests/host/rope_test.c` | `rope.h` |
| `tests/host/flags_test.c` | `flags.h` |
| `tests/host/engine_smoke.c` | `engine.h` |
| `tests/host/engine_features_test.c` | `engine.h` |
| `tests/host/par_golden.c` | `engine.h`, `kx.h` (private) |
| `tests/host/par_selfcheck.c` | `../cuda/parity_test.c` (textual `#include` — the host self-check reuses the GPU comparator source verbatim) |
| `tests/cuda/parity_test.c` | `engine.h`, `cuda_rt.h`, `kx.h` (private) |

## 2. Module-level dependencies (via public headers)

Direction: A → B means "A depends on B".

| Module (TU) | Depends on |
|-------------|-----------|
| core (`mimfer.c`) | tensor |
| config (`config.c`) | cuda_rt (portable surface), sampling |
| alloc (`alloc.c`) | — (leaf; `[cuda]` runtime only under `MM_WITH_CUDA`) |
| sampling (`sampling.c`) | — (leaf) |
| cuda_rt (`cuda_rt.c`) | — (leaf; host stubs or real CUDA per `MM_WITH_CUDA`) |
| cuda_mem (`cuda_mem.c`) | cuda_rt (transitively via `cuda_mem.h` → `kernels.h` → `plan.h` → `cuda_rt.h`) |
| plan (`plan.c`) | config, cuda_rt, kernels, **engine** (for the `mm_kcall`/descriptor types) |
| kv (`kv.c`) | cuda_rt, config, alloc |
| sched (`sched.c`) | — (leaf) |
| rope (`rope.c`) | config, kernels |
| flags (`flags.c`) | config |
| tensor_registry (`tensor_registry.c`, private header) | engine, tensor |
| kernels dispatcher (`kx.c`) | kernels, engine |
| CPU reference kernels (`cpu/cx.c`) | kernels, engine, tensor (+ private `kx.h`) |
| CUDA launch set (`cuda/cx.cu`) | kernels, engine, tensor, cuda_rt (+ private `kx.h`) |
| engine (`engine.c`) | **all public modules** (hub via `engine.h`: config, alloc, tensor, kernels, kv, plan, sched, artifact, tokenizer, sampling, telemetry, rope) + cuda_mem + tensor_registry (private) |
| artifact / tokenizer / telemetry | each only its own header (unintegrated — B2) |

Observations:

- **`engine.h` is the single hub.** Every engine-level TU (engine,
  dispatcher, both kernel sets, plan, tensor_registry) includes it, so
  any public-header change revalidates the whole engine surface — which
  is exactly what `$(HDRS)` prerequisites enforce.
- **No circular module dependencies.** `engine.h` includes `plan.h`
  while `plan.c` includes `engine.h`, but that is a TU→header
  relationship, not a header cycle; the header DAG is acyclic (§1).
- **The three unintegrated modules** (artifact/tokenizer/telemetry)
  have zero dependents beyond their own TU and `engine.h`'s umbrella
  include; they add no coupling while unregistered.

## 3. Cross-TU symbol dependencies (link level)

Header includes are not the whole story — these symbols cross TUs:

| Symbol(s) | Defined in | Called from | Build where both link |
|-----------|-----------|-------------|------------------------|
| `kx_op_*` (18 launchers) | `src/kernels/cpu/cx.c` (host) **or** `src/kernels/cuda/cx.cu` (GPU) — one per build | `mm_kx_invoke()` in `src/kernels/kx.c` | host build: cpu/cx.c · GPU build: cx.cu (this is why the B1 fix matters — cx.cu must be on the nvcc link line) |
| `kx_cpu_oppref` | `src/kernels/cpu/cx.c:668` (host builds) **and** a `#ifdef MM_WITH_CUDA` reference copy in `tests/cuda/parity_test.c` (GPU build only — B10) | `check_coverage()` in `tests/cuda/parity_test.c` (both branches), `tests/host/par_golden.c:363` | host builds: from `cpu/cx.c` (the test copy is guarded off) · GPU build: from the test's guarded copy — see note |
| `kx_cuda_oppref` | `src/kernels/cuda/cx.cu` (C linkage via `kx.h` guard) | `tests/cuda/parity_test.c` (MM_WITH_CUDA branch only) | GPU build only (unreferenced in host builds, so the host prototypes are link-safe) |

Note on `kx_cpu_oppref` in the GPU build (B10): `GPU_SRCS` excludes
`cpu/cx.c` — and it cannot be added back, because it defines the same
18 `kx_op_*` launchers that `cx.cu` defines (linking both would
multiply-define them). Yet the coverage mirror assertion
(`kx_cuda_oppref(op) == kx_cpu_oppref(op)`) needs the CPU-reference
vector inside the GPU binary. The fix is the `#ifdef MM_WITH_CUDA`
reference copy at the top of the coverage section of
`tests/cuda/parity_test.c` (identical 18-opcode table, `default: 0`,
must stay in sync with `cpu/cx.c`). In the host self-check the guard is
off, so that binary links the real definition from `cpu/cx.c` and the
host recipe is unchanged. The mirror assertion then verifies, inside
the GPU binary, that `cx.cu`'s launcher coverage matches the
CPU-reference coverage vector; the per-op golden comparison remains the
substantive cross-check.

## 4. The `MM_WITH_CUDA` split (one define, two behaviors)

The host build never defines `MM_WITH_CUDA`; the Makefile sets it
exactly in `NVCCFLAGS`. What changes:

| File | Host (no define) | GPU (`-DMM_WITH_CUDA`, nvcc) |
|------|------------------|-------------------------------|
| `include/mimfer/cuda_rt.h` | portable surface only (build-probe constants, no CUDA types) | additionally includes `<cuda_runtime.h>` and exposes raw `cudaStream_t` handles |
| `src/cuda/cuda_rt.c` | honest no-op host stubs | real streams, events, graph capture, error-checked API wrappers |
| `src/cuda/cuda_mem.c` | host stubs | device arenas + pinned I/O block |
| `src/config/config.c` | device probe via stubs | real device query |
| `src/alloc/alloc.c` | host allocator path | + CUDA-aware paths |
| `src/kernels/cuda/cx.cu` | not compiled | compiled as C++/CUDA (language by extension; **no `-x`** — CUDA 13.1's `-x` is a global last-value-wins option) |
| `tests/cuda/*` | compiled as C as the host self-check (via `par_selfcheck.c`'s textual include, without the define) | compiled as C by nvcc (by extension) with the define; `gpu_smoke.c` `#error`s without it |

Consequence (BUILD_AUDIT.md B5): the host build is hermetic — it needs
no CUDA toolkit at all (verified: the default host target has never
required the toolkit and the host suite is green; note this host now has
the CUDA 13.1 toolkit installed, used only by the opt-in CUDA builds).

## 5. External dependencies

| Dependency | Used by | Notes |
|------------|---------|-------|
| C library (libc) | everything | — |
| libmath (`-lm`) | `rope.c`, `cpu/cx.c`, `plan_test.c`, `rope_test.c`, `engine_features_test.c` | linked via `LDLIBS := -lm` on every host binary |
| CUDA runtime (linked by nvcc) | GPU builds only | `cx.cu`, `cuda_rt.c`/`cuda_mem.c` real branches, `tests/cuda/*` |
| GPU + driver | execution only | RTX PRO 4000 Blackwell (sm_120) target; soft-gate affordance for other cards |

No vendored third-party code, no framework, no generated files
(`make distclean` removes everything `make` produces: `build/`).

