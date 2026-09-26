# DFlash2 speculative decoding — declared scope (mimfer)

Status: **planned work; validated scaffolding only.** This document is the
single source of truth for what the speculative-decoding surface does and
does not do in the current tree. The engine refuses to start with a
non-off backend (`MM_ERR_UNSUPPORTED`); nothing here is silently
non-functional.

## What exists (verified)

- **Flag surface** (`src/flags/flags.c`, `include/mimfer/flags.h`):
  `--spec off|mtp|dflash2`, `--draft-tokens` (0..8), `--lm-head-draft`.
  Every value is parsed and range-checked; cross-field rules live in
  `mm_engine_cfg_validate` (`src/config/config.c`): `--spec != off` requires
  `--draft-tokens` in 1..8, and `--lm-head-draft` requires `--spec != off`.
- **Profile metadata** (`src/config/config.c`, `MM_PROFILES`): each weights
  profile records the draft window its artifact carries
  (`dflash2_max`: 8 for `quasar`, 4 for `neroued`) and the MTP layer count;
  the draft-window ceiling is validated against the profile when both are
  set.
- **Engine-level refusal** (`src/engine/engine.c`, `mm_engine_load`):
  `--spec mtp` or `--spec dflash2` starts with an explicit error
  ("start with --spec off") and `MM_ERR_UNSUPPORTED`. The warning is logged
  so an operator sees exactly why the process did not start.
- **Test coverage**: `tests/host/flags_test.c` (parse + cross-field
  validation) and `tests/host/engine_features_test.c` (engine-level
  refusal, host + engine smoke of the off path).

## What does NOT exist

- No draft/verify execution loop (no draft-token generation, no verify
  step, no accept-length handling) in `mm_engine_step`.
- No draft LM head path (`--lm-head-draft` validates but selects nothing;
  the MTP head in the artifact is not wired into the forward pass).
- No DFlash2 window machinery (the `dflash2_max` ceiling is metadata +
  validation only; there is no windowed draft kernel set).

## Planned shape (intent, not commitment)

When implemented, the draft/verify loop will attach to the existing
validated architecture per the project rules (READ_MEMORY.md §3/§7):

1. a small self-contained module (draft proposal + verify in one step),
2. one registration in the build,
3. one integration point in the step loop (`src/engine/engine.c`),
4. the CPU reference first, the parity gate re-run before the CUDA path is
   touched, and the refusal in `mm_engine_load` removed only when
   `VALIDATION_CHECKLIST.md` passes for the speculative path.

Nothing in this document claims support for speculative decoding today:
the supported configuration is `--spec off`.
