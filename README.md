# mimfer

mimfer is a small C inference appliance for a single NVIDIA GPU: one Linux
process, one CUDA device, one resident model. The core is C11 — no C++, no
exceptions, no third-party headers — and CUDA enters only through
`<cuda_runtime.h>` inside `src/cuda/` and `src/kernels/`. Execution is
graph-captured, and a CPU reference kernel set acts as the correctness oracle
for the GPU kernels (per-op parity checks with explicit tolerance classes).

Mimfer is an independent implementation inspired by the engineering
philosophy behind several Ninfer-related projects and suckless software, but
it is not a fork and does not reuse their source code.

**Status.** The host (CPU) path is implemented and verified (planner, CPU
kernels, engine, parity self-check, ASan/UBSan clean, deterministic across
runs). The CUDA path is written but has **not** been compiled with `nvcc` or
executed on a real GPU. Until `VALIDATION_CHECKLIST.md` passes on hardware,
no performance or compatibility claim about the GPU path should be taken.

## Current Model Support

Mimfer is currently designed, developed, and validated only against the
following models:

- `MirkoCovizzi/Qwen3.8-27B-QUASAR-NVFP4-NInfer`
- `Neroued/Qwen3.8-27B-nvfp4-NInfer`

These are currently the primary reference models. Other models are **not**
officially supported yet, and no other architectures have been validated. The
engine limits (sequence length, decode slots, KV paging) are fixed at
compile time for this one workload shape, and nothing in the codebase is
intended to claim generic model compatibility.

Support for other model families may be added later (see "Extensibility").
Until it is validated on the reference hardware, treat any other model as
unsupported.

## Current Artifact Support

Mimfer supports:

- **NInfer Artifact V2**
- **NInfer Artifact V3**

The container format (`include/mimfer/artifact.h`,
`src/artifact/artifact.c`) is organized around the two families: the base
sections (model config, tokenizer, chat template, packed NVFP4 weights, MTP
head, generation config), the V2 family (prompt template, context bundle,
cached conversation, reusable session) and the V3 family (structured memory,
workflow graph, agent state, execution checkpoint, reasoning metadata, task
bundle). Versioning is explicit: a major/minor format pair plus a capability
word, with forward-skippable and backward-rejecting rules.

Current development and validation use the existing NInfer artifact
ecosystem (NVFP4-quantized `qwen3.8-27b` artifacts). Mimfer aims to remain
compatible with those formats. Future extensions may introduce
Mimfer-specific capabilities, but they would be additive, versioned, and
would not break reading of existing V2/V3 artifacts.

One caveat: the artifact reader is implemented and self-contained, but
in the current engine builds the loader is not yet wired into the lifecycle
(the engine tests run a built-in standing-in model shape — see
`RELEASE_READINESS.md` §3). The format support is real; full end-to-end
artifact-driven runs are still pending.

## Hardware Scope

Current development and testing target a single hardware configuration:

- **NVIDIA RTX PRO 4000 Blackwell** (sm_120, 24 GiB GDDR7; CUDA 12.8 or
  newer)

This is currently the primary development platform, and it is the **only**
hardware configuration targeted during development. Performance
characteristics outside this configuration are currently unknown, and no
support is claimed for hardware that has not been validated.

The engine enforces the device profile at startup as a hard gate. A soft-gate
environment variable (`MIMFER_SOFT_DEVICE_GATE`) allows starting on other
NVIDIA GPUs for validation work — that is a testing affordance, not a support
statement.

## Extensibility

The design is inspired by the simplicity philosophy of suckless software —
as an architectural influence, not as a code dependency (nothing from those
projects is linked or vendored here).

The core project intentionally stays small, focused, and understandable.
Rather than creating a massive framework with dozens of abstraction layers,
the design goal is that new functionality should be easy to add directly.
Examples of what this is meant to cover:

- support for new GPUs
- support for new model families
- support for new artifact formats
- alternative samplers
- additional kernels
- specialized runtimes

In most cases that should require:

- a small self-contained module,
- registration in the build (single-line `gcc`/`nvcc` commands — there is no
  Makefile or CMake to keep in sync),
- a minimal integration point in the runtime.

The goal is that a developer can add a feature by introducing focused code
and wiring it into the engine, rather than modifying large parts of the
system. What the codebase is trying to keep:

- explicit design (no hidden registries, no implicit magic)
- low complexity (few moving parts, short call paths)
- readable code (C that reads top to bottom)
- modular components (each subsystem owns one job; see `READ_MEMORY.md` §3)
- simple integration (one call site per feature)

## Documentation

| File | What it is |
|------|------------|
| `READ_MEMORY.md` | Project state and architecture notes (read first when touching code) |
| `GPU_VALIDATION.md` | Hardware validation runbook: exact build/run/test procedure for a real GPU |
| `VALIDATION_CHECKLIST.md` | Go/no-go checklist for the first GPU run |
| `RELEASE_READINESS.md` | Completed vs. pending subsystems, risks, unverified assumptions |

There is no build system: every build is a single `gcc` or `nvcc` command
run from the repository root (commands, flags, and expected outputs are in
`GPU_VALIDATION.md`).
