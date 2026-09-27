# ============================================================================
# mimfer — CMake/Ninja build system (root, single file)
#
# Specified commands:
#   cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
#   cmake --build build -j
#
# Produces the same artifacts as the root Makefile (GNU Make; both systems
# are kept and equivalent), all in build/ (= CMAKE_BINARY_DIR):
#
#   host (gcc):  plan_test  rope_test  flags_test  device_gate_test
#                artifact_test  engine_smoke  engine_features
#                slot_lifecycle_test  par_golden  par_selfcheck
#   GPU  (nvcc): parity_test  gpu_smoke  gpu_slot   (-DMIMFER_ENABLE_CUDA=ON)
#
# Test targets (equivalents of the Makefile's run targets):
#   cmake --build build --target test     == make test
#   cmake --build build --target parity   == make parity
#   cmake --build build --target smoke    == make smoke
#   cmake --build build --target golden   == make golden
#   cmake --build build --target slot-lifecycle == make slot-lifecycle
#
# Flag policy (do NOT change without re-running the parity gate —
# GPU_VALIDATION.md §3.4):
#   * host (gcc):  -std=c11 -Wall -Wextra -Werror, includes
#     -Iinclude -Isrc/model -Isrc/kernels, link -lm. Deliberately NO
#     optimizer or debug flags: the CPU path is the golden reference and the
#     parity golden is tied to the bit-determinism of this exact recipe.
#     For that reason the CMAKE_BUILD_TYPE flag suffixes (Release's
#     -O3 -DNDEBUG, ...) are cleared below — the build type changes nothing
#     for mimfer host code; it is accepted only so the specified command
#     configures cleanly.
#   * GPU (nvcc):  -O2 -fmad=false -DMM_WITH_CUDA, arch from
#     MIMFER_CUDA_ARCH (default sm_120, the RTX PRO 4000 Blackwell;
#     Makefile: CUDA_ARCH ?= sm_120). -fmad=false is MANDATORY while the
#     CPU oracle is FMA-free. Every .c translation unit is compiled by nvcc
#     AS C (-x c): the dispatcher ABI is pinned to C linkage (hard
#     invariant, GPU_VALIDATION.md §3.4/§11). cx.cu is the only CUDA (C++)
#     translation unit and gets no -x.
#
# CUDA builds are OFF by default so that host-only machines (no nvcc /
# CUDA toolkit) configure and build cleanly:
#   cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
#         -DMIMFER_ENABLE_CUDA=ON [-DMIMFER_CUDA_ARCH=sm_120]
# requires nvcc on PATH (or -DCMAKE_CUDA_COMPILER=/path/to/nvcc) and the
# CUDA toolkit (>= 12.8); pin the toolkit root with
# -DCMAKE_CUDA_TOOLKIT_ROOT_DIR=/path/to/cuda if it is not found via nvcc.
# GPU *compilation* itself is still pending first validation on a GPU
# machine — same status as the Makefile's 'make cuda'.
# ============================================================================

cmake_minimum_required(VERSION 3.20)
project(mimfer VERSION 0.1.0 LANGUAGES C)

# ---- options / variables ----------------------------------------------------

option(MIMFER_ENABLE_CUDA
    "Build the CUDA (GPU) test binaries (requires nvcc + CUDA toolkit >= 12.8)"
    OFF)
set(MIMFER_CUDA_ARCH "sm_120" CACHE STRING
    "CUDA architecture for the GPU builds (Makefile: CUDA_ARCH ?= sm_120). Override per card, e.g. sm_90.")
set(MIMFER_NVCC_EXTRA "" CACHE STRING
    "Extra nvcc flags, appended after the NVCCFLAGS equivalent (Makefile: NVCC_EXTRA ?=). On a machine without a GPU driver: -DMIMFER_NVCC_EXTRA=-L<cuda toolkit>/lib64/stubs")

# MIMFER_CUDA_ARCH validation (Makefile: CUDA_ARCH ?= sm_120). The arch goes
# straight into nvcc's -arch flag in the GPU custom commands below.
if(MIMFER_ENABLE_CUDA)
    if(NOT MIMFER_CUDA_ARCH MATCHES "^sm_[0-9]+$")
        message(FATAL_ERROR
            "MIMFER_CUDA_ARCH must be of the form sm_NNN (e.g. sm_120), "
            "got: ${MIMFER_CUDA_ARCH}")
    endif()
endif()

# ---- host flags (must stay exact — see flag policy above) ---------------------

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)          # -std=c11, not -std=gnu11

string(APPEND CMAKE_C_FLAGS " -Wall -Wextra -Werror")

# The Makefile recipe carries no optimizer/debug flags and the parity golden
# is tied to its bit-determinism: clear the per-build-type flag suffixes so
# CMAKE_BUILD_TYPE=Release does not add -O3/-DNDEBUG to the mimfer host code.
set(CMAKE_C_FLAGS_RELEASE "")
set(CMAKE_C_FLAGS_MINSIZEREL "")
set(CMAKE_C_FLAGS_DEBUG "")
set(CMAKE_C_FLAGS_RELWITHDEBINFO "")

# Include paths — identical set for every binary, as in the Makefile:
#   include     — public headers (mimfer/*.h)
#   src/model   — tensor_registry.h (included by src/engine/engine.c)
#   src/kernels — kx.h (included by src/kernels/cpu/cx.c)
include_directories(include src/model src/kernels)

# ---- source sets (mirror the Makefile variable names) -------------------------

# Planner-only set (plan_test): no engine, no kernel dispatcher.
set(PLAN_SRCS
    src/plan/plan.c
    src/alloc/alloc.c
    src/config/config.c
    src/core/mimfer.c
    src/cuda/cuda_rt.c
    src/sampling/sampling.c)

# Full engine set (every engine-linked host binary).
set(ENGINE_SRCS
    src/engine/engine.c
    src/plan/plan.c
    src/alloc/alloc.c
    src/config/config.c
    src/core/mimfer.c
    src/cuda/cuda_rt.c
    src/cuda/cuda_mem.c
    src/sampling/sampling.c
    src/kv/kv.c
    src/kernels/cpu/cx.c
    src/kernels/kx.c
    src/model/tensor_registry.c
    src/rope/rope.c
    src/sched/sched.c)

# I/O modules (Makefile: IO_SRCS): the .mimfer artifact container, the
# byte-BPE tokenizer and the telemetry ring. Linked into every engine
# binary (the engine owns the artifact handle, the embedded tokenizer and
# the telemetry ring) and the artifact fixture test below.
set(IO_SRCS
    src/artifact/artifact.c
    src/tokenizer/tokenizer.c
    src/telemetry/telemetry.c)
list(APPEND ENGINE_SRCS ${IO_SRCS})

# Small unit-test sets (no engine): RoPE tables / CLI flag grammar.
set(CFG_TEST_SRCS
    src/config/config.c
    src/core/mimfer.c
    src/sampling/sampling.c)

# ---- host binaries (same names and link sets as the Makefile) ------------------

add_executable(plan_test       tests/host/plan_test.c ${PLAN_SRCS})
add_executable(engine_smoke    tests/host/engine_smoke.c ${ENGINE_SRCS})
add_executable(rope_test       tests/host/rope_test.c src/rope/rope.c ${CFG_TEST_SRCS})
add_executable(flags_test      tests/host/flags_test.c src/flags/flags.c ${CFG_TEST_SRCS})
# Device gate (mm_device_check): VRAM tolerance regression-anchored to the
# measured driver value on silicon (23.4256 GiB on the 24 GB card).
add_executable(device_gate_test tests/host/device_gate_test.c ${CFG_TEST_SRCS})
# Artifact/tokenizer/telemetry fixture test: in-process .mimfer fixtures,
# container integrity + corruption cases, byte-BPE encode/decode, mbuf
# codec, telemetry ring, and the engine's real --artifact path.
add_executable(artifact_test tests/host/artifact_test.c ${ENGINE_SRCS})
add_executable(engine_features tests/host/engine_features_test.c ${ENGINE_SRCS})
# Slot lifecycle regression (multi-request teardown): one shared source,
# built with gcc here and by nvcc for the GPU binary gpu_slot below.
add_executable(slot_lifecycle_test tests/host/slot_lifecycle_test.c ${ENGINE_SRCS})
add_executable(par_golden      tests/host/par_golden.c ${ENGINE_SRCS})
# Host self-check: thin wrapper that #includes tests/cuda/parity_test.c
# (one shared comparator, built without MM_WITH_CUDA).
add_executable(par_selfcheck   tests/host/par_selfcheck.c ${ENGINE_SRCS})

set(MIMFER_HOST_BINS
    plan_test engine_smoke rope_test flags_test device_gate_test
    artifact_test engine_features slot_lifecycle_test par_golden
    par_selfcheck)

# Every Makefile host binary links -lm (no CUDA target may receive it).
foreach(bin IN LISTS MIMFER_HOST_BINS)
    target_link_libraries(${bin} PRIVATE m)
endforeach()

# Build all host binaries (== 'make host' / bare 'make').
add_custom_target(host DEPENDS ${MIMFER_HOST_BINS})

# ---- host test targets (equivalents of the Makefile's run targets) --------------
#
# Each 'run' target builds its binary (DEPENDS) and then executes it, exactly
# like the matching Makefile target. 'test' runs the full Makefile sequence
# (plan → rope → flags → device-gate → engine-smoke → engine-features →
# golden → parity self-check) via scripts/run_host_tests.sh, which makes the
# suite sequential and fail-fast under any parallelism.
#
#   cmake --build build --target test     == make test
#   cmake --build build --target parity   == make parity
#   cmake --build build --target smoke    == make smoke

set(MIMFER_GOLDEN_BIN ${CMAKE_BINARY_DIR}/par_golden.bin)

# Golden writer. Always runs when requested (like the phony Makefile
# 'golden' target): the golden is rewritten before every parity self-check,
# so it is never compared against a stale golden (kernel changes re-gate).
add_custom_target(golden
    COMMAND $<TARGET_FILE:par_golden> ${MIMFER_GOLDEN_BIN}
    DEPENDS par_golden
    COMMENT "writing parity golden build/par_golden.bin (CPU oracle)")

add_custom_target(plan-test
    COMMAND $<TARGET_FILE:plan_test>
    DEPENDS plan_test
    COMMENT "== plan_test (host)")

add_custom_target(rope-test
    COMMAND $<TARGET_FILE:rope_test>
    DEPENDS rope_test
    COMMENT "== rope_test (host)")

add_custom_target(flags-test
    COMMAND $<TARGET_FILE:flags_test>
    DEPENDS flags_test
    COMMENT "== flags_test (host)")

add_custom_target(device-gate-test
    COMMAND $<TARGET_FILE:device_gate_test>
    DEPENDS device_gate_test
    COMMENT "== device_gate_test (host)")

add_custom_target(artifact-test
    COMMAND $<TARGET_FILE:artifact_test>
    DEPENDS artifact_test
    COMMENT "== artifact_test (host)")

add_custom_target(engine-smoke
    COMMAND $<TARGET_FILE:engine_smoke>
    DEPENDS engine_smoke
    COMMENT "== engine_smoke (host)")

add_custom_target(engine-features
    COMMAND $<TARGET_FILE:engine_features>
    DEPENDS engine_features
    COMMENT "== engine_features (host)")

add_custom_target(slot-lifecycle
    COMMAND $<TARGET_FILE:slot_lifecycle_test>
    DEPENDS slot_lifecycle_test
    COMMENT "== slot_lifecycle_test (host)")

# Host parity self-check (oracle pre-gate): expect PARITY PASSED.
add_custom_target(parity-selfcheck
    COMMAND $<TARGET_FILE:par_selfcheck> ${MIMFER_GOLDEN_BIN}
    DEPENDS par_selfcheck golden
    COMMENT "== parity self-check (host oracle gate)")

# Strict sequential host test suite (== 'make test').
#
# Implemented as a repo script (scripts/run_host_tests.sh) invoked with the
# build dir as its single argument, NOT as an embedded shell sequence, for a
# concrete reason: ninja executes a target's command line through the shell
# (`sh -c "<raw command line>"`), and CMake's Ninja generator escapes every
# space INSIDE a command argument with a backslash. The outer shell then
# re-parses that escaped script, so backslash-spaces become literal in-word
# spaces and the sequence falls apart (words get glued, `;`-terminated
# fragments turn into single bogus commands, path words gain stray spaces
# and fail with 'not found'). A script file keeps every command token
# single-word — `sh <script> <builddir>` — so the sequence is exactly the
# script's, sequential and fail-fast, under any parallelism.
add_custom_target(test
    COMMAND /bin/sh ${CMAKE_SOURCE_DIR}/scripts/run_host_tests.sh ${CMAKE_BINARY_DIR}
    DEPENDS ${MIMFER_HOST_BINS}
    COMMENT "HOST TEST SUITE (plan_test, rope_test, flags_test, device_gate_test, "
            "artifact_test, engine_smoke, engine_features, slot_lifecycle_test, "
            "parity self-check)")

# Aliases (== 'make parity', 'make smoke', 'make tests').
add_custom_target(parity DEPENDS parity-selfcheck)
add_custom_target(smoke  DEPENDS engine-smoke)
add_custom_target(tests  DEPENDS test)

# ---- CUDA (GPU) builds — optional, OFF by default -------------------------------
#
# Design (why custom commands, not CMake CUDA targets): CMake's per-source
# LANGUAGE property is PROJECT-GLOBAL. Marking the shared .c sources "CUDA"
# so nvcc compiles them for the GPU binaries would force nvcc on the same
# files in every host-only binary too, breaking the host golden path (the
# Makefile compiles those with $(CC), never nvcc). The GPU binaries are
# therefore built with ONE nvcc invocation each — the Makefile recipe
# (Makefile:159-166) verbatim. enable_language(CUDA) is used only to locate
# nvcc and the toolkit (CUDAToolkit_VERSION for the >= 12.8 gate); no CMake
# target uses the CUDA language, so nothing is compiled through it and the
# host build stays 100% C.

if(MIMFER_ENABLE_CUDA)
    enable_language(CUDA)

    if(DEFINED CUDAToolkit_VERSION AND CUDAToolkit_VERSION VERSION_LESS 12.8)
        message(WARNING
            "CUDA toolkit ${CUDAToolkit_VERSION} is older than the "
            "validated 12.8 (GPU_VALIDATION.md §1)")
    endif()

    # Makefile: NVCCFLAGS := -O2 -fmad=false -DMM_WITH_CUDA -arch=$(CUDA_ARCH)
    #            NVCCFLAGS += -Iinclude -Isrc/model -Isrc/kernels
    #   -O2            the verified toolchain flag; keep for comparability
    #   -fmad=false    mandatory while the CPU oracle is FMA-free
    #   -DMM_WITH_CUDA enables the real CUDA paths
    #   -arch          passed directly (Makefile:46), no gencode expansion
    # The -I paths are absolute: the Makefile runs nvcc from the source root,
    # the custom command runs from the build directory.
    # -lcuda: mm_device_probe() queries the memory clock rate through the
    # driver API (cuDeviceGetAttribute), which replaced the cudaDeviceProp
    # field in CUDA 13. Link-only: codegen (parity gate) is untouched.
    # On a machine without a GPU driver, the linker needs
    # -L<CUDA toolkit>/lib64/stubs (see Makefile NVCCFLAGS note).
    set(MIMFER_NVCCFLAGS
        -O2 -fmad=false -DMM_WITH_CUDA -arch=${MIMFER_CUDA_ARCH}
        -I${CMAKE_CURRENT_SOURCE_DIR}/include
        -I${CMAKE_CURRENT_SOURCE_DIR}/src/model
        -I${CMAKE_CURRENT_SOURCE_DIR}/src/kernels
        -lcuda)

    # GPU set (Makefile:89): the engine sources without the CPU reference
    # kernels, plus the CUDA launch set. NO -x flags: nvcc chooses the
    # language by extension (.c -> C, .cu -> CUDA). In CUDA 13.1 -x is a
    # GLOBAL last-value-wins option (nvcc warns "incompatible redefinition
    # for option 'x'"), so any -x on the line forces one language on ALL
    # files — the old per-source "-x c" approach compiled cx.cu as plain C
    # (no __global__, no <<<>>>). cx.cu defines the kx_op_* launchers that
    # replace cpu/cx.c — omitting it is a hard link failure (Makefile:95).
    set(MIMFER_GPU_C_SRCS
        src/engine/engine.c
        src/plan/plan.c
        src/alloc/alloc.c
        src/config/config.c
        src/core/mimfer.c
        src/cuda/cuda_rt.c
        src/cuda/cuda_mem.c
        src/sampling/sampling.c
        src/kv/kv.c
        src/kernels/kx.c
        src/model/tensor_registry.c
        src/rope/rope.c
        src/sched/sched.c
        src/artifact/artifact.c
        src/tokenizer/tokenizer.c
        src/telemetry/telemetry.c)
    set(MIMFER_GPU_CU_SRCS src/kernels/cuda/cx.cu)

    # Header deps (Makefile:99-100): one-shot builds, no .d tracking.
    file(GLOB MIMFER_GPU_HDRS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        "${CMAKE_CURRENT_SOURCE_DIR}/include/mimfer/*.h")
    list(APPEND MIMFER_GPU_HDRS src/kernels/kx.h src/model/tensor_registry.h)

    # One nvcc command per binary — the Makefile:160-166 recipes. Language
    # per file by extension (no -x; see the note above).
    set(MIMFER_GPU_C_ABS "")
    foreach(src IN LISTS MIMFER_GPU_C_SRCS)
        list(APPEND MIMFER_GPU_C_ABS "${CMAKE_CURRENT_SOURCE_DIR}/${src}")
    endforeach()
    foreach(bin parity_test gpu_smoke)
        add_custom_command(
            OUTPUT ${CMAKE_BINARY_DIR}/${bin}
            COMMAND ${CMAKE_CUDA_COMPILER} ${MIMFER_NVCCFLAGS} ${MIMFER_NVCC_EXTRA}
                    "${CMAKE_CURRENT_SOURCE_DIR}/tests/cuda/${bin}.c"
                    ${MIMFER_GPU_C_ABS}
                    "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/cuda/cx.cu"
                    -o ${CMAKE_BINARY_DIR}/${bin}
            DEPENDS "tests/cuda/${bin}.c"
                    ${MIMFER_GPU_C_SRCS} ${MIMFER_GPU_CU_SRCS}
                    ${MIMFER_GPU_HDRS}
            COMMENT "== nvcc ${bin} (arch ${MIMFER_CUDA_ARCH})")
    endforeach()

    # Slot lifecycle regression (GPU build): the SAME source as the host
    # binary slot_lifecycle_test (tests/host/slot_lifecycle_test.c), compiled
    # by nvcc as C against the GPU engine set (Makefile: GPU_SLOT).
    add_custom_command(
        OUTPUT ${CMAKE_BINARY_DIR}/gpu_slot
        COMMAND ${CMAKE_CUDA_COMPILER} ${MIMFER_NVCCFLAGS} ${MIMFER_NVCC_EXTRA}
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/host/slot_lifecycle_test.c"
                ${MIMFER_GPU_C_ABS}
                "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/cuda/cx.cu"
                -o ${CMAKE_BINARY_DIR}/gpu_slot
        DEPENDS "tests/host/slot_lifecycle_test.c"
                ${MIMFER_GPU_C_SRCS} ${MIMFER_GPU_CU_SRCS}
                ${MIMFER_GPU_HDRS}
        COMMENT "== nvcc gpu_slot (arch ${MIMFER_CUDA_ARCH})")

    set(MIMFER_CUDA_BINS
        ${CMAKE_BINARY_DIR}/parity_test
        ${CMAKE_BINARY_DIR}/gpu_smoke
        ${CMAKE_BINARY_DIR}/gpu_slot)

    # Build both GPU binaries (== 'make cuda'; build only, never run).
    add_custom_target(cuda DEPENDS ${MIMFER_CUDA_BINS})
else()
    message(STATUS
        "MIMFER_ENABLE_CUDA=OFF — host build only "
        "(add -DMIMFER_ENABLE_CUDA=ON for the GPU binaries)")
endif()
