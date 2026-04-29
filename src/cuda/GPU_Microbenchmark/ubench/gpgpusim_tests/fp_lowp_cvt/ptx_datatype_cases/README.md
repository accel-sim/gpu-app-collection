# PTX Datatype Parser Smoke Cases

`run_ptx_datatype_parser_smoke.sh` validates lexer/parser + IR construction only.
Does not execute PTX instructions, so `instructions.cc:cvt_impl()` runtime
bounds checking for `g_cvt_fn[src_fmt][dst_fmt]` cannot be exercised here.

The cvt bounds guard remains in runtime code to protect execution paths that
use datatype format IDs beyond the legacy conversion table dimensions.

`run_lowp_cvt_ptxas_crosscheck.sh` includes a spec-tracking gate for
`cvt.*.ue8m0x2.bf16x2` with `.relu`. By default
`LOWP_CVT_SPEC_TRACK_UE8M0X2_BF16X2_RELU=auto` (track current toolchain
behavior); set `LOWP_CVT_SPEC_TRACK_UE8M0X2_BF16X2_RELU=ptxas_accepts` when
toolchains begin accepting the form.
