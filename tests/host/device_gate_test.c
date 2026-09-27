/*
 * Host test for the device gate (src/config/config.c, mm_device_check):
 * name / compute capability / VRAM / derived-bandwidth checks against the
 * MM_TARGET_PRO4000 profile.
 *
 * The VRAM check is regression-anchored to a value MEASURED ON SILICON
 * (RTX PRO 4000 Blackwell, driver 595.71.05, 2026-09-27): the driver
 * reports totalGlobalMem = 25,153,044,480 bytes = 23.4256 GiB on the 24 GB
 * card — 2.39% below nominal because the driver/firmware reserves part of
 * the card. Before the tolerance fix the gate rejected the target card
 * itself (23 GiB < 24 GiB) and every GPU run needed the
 * MIMFER_SOFT_DEVICE_GATE bypass. The gate must accept the measured value
 * (within MM_VRAM_TOLERANCE_PCT of nominal) and reject a card that falls
 * more than the tolerance short.
 *
 * The checks are pure (no CUDA needed): mm_device_check takes a probed
 * mm_dev_info and the target profile.
 */
#include "mimfer/config.h"

#include <stdio.h>
#include <string.h>

static int g_fail = 0;
#define CHECK(cond, msg)                                                      \
    do {                                                                      \
        if (!(cond)) {                                                        \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, msg);     \
            g_fail = 1;                                                       \
        }                                                                     \
    } while (0)

/* Driver-reported totalGlobalMem on the real card (measured 2026-09-27,
 * driver 595.71.05, both devices of the pair). */
#define MEASURED_TOTAL_GLOBAL_MEM ((size_t)25153044480)

/* A probed device that IS the target card, using the measured driver
 * values where they were measured. */
static mm_dev_info probed(void)
{
    mm_dev_info d;
    memset(&d, 0, sizeof d);
    snprintf(d.name, sizeof d.name, "NVIDIA RTX PRO 4000 Blackwell");
    d.cc_major = 12;
    d.cc_minor = 0;
    d.sm_count = 70;
    d.vram_bytes = MEASURED_TOTAL_GLOBAL_MEM;
    d.l2_bytes = 96 * 1024 * 1024;
    d.smem_per_sm = 100 * 1024;
    d.bw_gbps = 672;
    d.max_threads_per_sm = 1536;
    d.regs_per_sm = 65536;
    return d;
}

static void test_real_card_passes(void)
{
    mm_dev_info d = probed();
    CHECK(mm_device_check(&d, &MM_TARGET_PRO4000) == MM_OK,
          "measured driver values (23.4256 GiB, 2.39% below nominal) pass");
}

static void test_vram_tolerance(void)
{
    size_t nominal = MM_TARGET_PRO4000.vram_bytes;    /* 24 GiB */
    size_t floor = nominal * (100 - MM_VRAM_TOLERANCE_PCT) / 100;
    mm_dev_info d;

    /* Exactly at the tolerance floor: inclusive boundary. */
    d = probed();
    d.vram_bytes = floor;
    CHECK(mm_device_check(&d, &MM_TARGET_PRO4000) == MM_OK,
          "at the tolerance floor the gate passes (inclusive)");

    /* One byte below the floor: rejected. */
    d = probed();
    d.vram_bytes = floor - 1;
    CHECK(mm_device_check(&d, &MM_TARGET_PRO4000) == MM_ERR_DEVICE,
          "one byte below the tolerance floor is rejected");

    /* Above nominal (a bigger card with the matching name) passes —
     * the gate is a lower bound. */
    d = probed();
    d.vram_bytes = nominal + (nominal / 4);
    CHECK(mm_device_check(&d, &MM_TARGET_PRO4000) == MM_OK,
          "above nominal passes");

    /* The measured value really is inside the tolerance band. */
    d = probed();
    CHECK(d.vram_bytes >= floor && d.vram_bytes < nominal,
          "measured value sits inside the tolerance band");
}

static void test_name_gate(void)
{
    mm_dev_info d = probed();
    snprintf(d.name, sizeof d.name, "NVIDIA RTX PRO 5000 Blackwell");
    CHECK(mm_device_check(&d, &MM_TARGET_PRO4000) == MM_ERR_DEVICE,
          "a non-matching card name is rejected");
}

static void test_cc_gate(void)
{
    mm_dev_info d;

    d = probed();
    d.cc_major = 9; d.cc_minor = 0;
    CHECK(mm_device_check(&d, &MM_TARGET_PRO4000) == MM_ERR_DEVICE,
          "cc 9.0 is rejected");

    d = probed();
    d.cc_major = 12; d.cc_minor = 1;
    CHECK(mm_device_check(&d, &MM_TARGET_PRO4000) == MM_OK,
          "cc 12.1 is accepted");

    d = probed();
    d.cc_major = 13; d.cc_minor = 0;
    CHECK(mm_device_check(&d, &MM_TARGET_PRO4000) == MM_ERR_DEVICE,
          "cc 13.0 is rejected");
}

static void test_bw_gate(void)
{
    mm_dev_info d;

    /* -10% boundary: inclusive. */
    d = probed();
    d.bw_gbps = MM_TARGET_PRO4000.bw_gbps * 9 / 10;
    CHECK(mm_device_check(&d, &MM_TARGET_PRO4000) == MM_OK,
          "derived bandwidth at -10% passes (inclusive)");

    /* +10% boundary plus one: rejected. */
    d = probed();
    d.bw_gbps = MM_TARGET_PRO4000.bw_gbps * 11 / 10 + 1;
    CHECK(mm_device_check(&d, &MM_TARGET_PRO4000) == MM_ERR_DEVICE,
          "derived bandwidth above +10% is rejected");

    /* Far off nominal: rejected. */
    d = probed();
    d.bw_gbps = 500;
    CHECK(mm_device_check(&d, &MM_TARGET_PRO4000) == MM_ERR_DEVICE,
          "derived bandwidth far off nominal is rejected");
}

int main(void)
{
    test_real_card_passes();
    test_vram_tolerance();
    test_name_gate();
    test_cc_gate();
    test_bw_gate();
    if (g_fail) {
        fprintf(stderr, "DEVICE GATE TEST FAILED\n");
        return 1;
    }
    printf("DEVICE GATE TEST PASSED\n");
    return 0;
}
