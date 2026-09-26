/*
 * CLI parsing: the server flag surface.
 *
 * One place owns the flag grammar: mm_flags_parse maps argv onto
 * mm_engine_cfg (the engine's only user-visible knobs, config.h) and
 * validates it (mm_engine_cfg_validate). Every flag is real: parsed,
 * range-checked, listed in mm_flags_help, and either changing engine
 * behavior or failing with an explicit error. Planned backends
 * (speculative decoding: docs/dflash2.md) validate their flags and are
 * then refused at engine start with MM_ERR_UNSUPPORTED.
 *
 * Grammar: long flags only, "--name value"; boolean flags take no value.
 * A weights profile (--weights-profile) applies its context defaults
 * unless the corresponding flag was given explicitly.
 */
#ifndef MIMFER_FLAGS_H
#define MIMFER_FLAGS_H

#include "config.h"

/* Help text budget (all flags, one line each). */
#define MM_FLAGS_HELP_SZ 8192

/* Parse argv (argv[0] = program name) into *cfg, which is defaulted
 * first (mm_engine_cfg_default) and then overwritten by the flags.
 * *help (may be NULL) is set to 1 when --help/-h is seen (the help text
 * is printed and *cfg is left at defaults); otherwise 0. Returns MM_OK
 * on success, MM_ERR_RANGE for unknown/malformed flags or out-of-range
 * values, MM_ERR_STATE for a missing flag value — each with a logged
 * message. */
mm_status mm_flags_parse(int argc, char **argv, mm_engine_cfg *cfg, int *help);

/* Write the help text (every flag, one line each) into buf of size n;
 * returns the length needed excluding the NUL. */
size_t mm_flags_help(char *buf, size_t n);

#endif /* MIMFER_FLAGS_H */
