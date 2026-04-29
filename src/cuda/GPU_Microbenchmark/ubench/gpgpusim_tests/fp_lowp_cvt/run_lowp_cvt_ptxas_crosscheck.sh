#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <ptx-cases-root>" >&2
  exit 2
fi

cases_root="$1"

if ! command -v ptxas >/dev/null 2>&1; then
  echo "ptxas not found; lowp cvt ptxas cross-check cannot run" >&2
  exit 1
fi

work_dir="$(mktemp -d)"
trap 'rm -rf "$work_dir"' EXIT

failures=0
ran=0

ptx_target() {
  awk '$1 == ".target" {print $2; exit}' "$1"
}

run_expect_pass() {
  local rel="$1"
  local file="$cases_root/$rel"
  local arch
  local out="$work_dir/$(basename "$file").out"
  local cubin="$work_dir/$(basename "$file").cubin"

  ((ran += 1))
  arch="$(ptx_target "$file")"
  if [[ -z "$arch" ]]; then
    echo "[FAIL] missing .target in $file"
    ((failures += 1))
    return
  fi

  if ptxas -arch="$arch" "$file" -o "$cubin" >"$out" 2>&1; then
    echo "[PASS] ptxas accepted: $rel"
  else
    echo "[FAIL] expected ptxas success: $rel"
    cat "$out"
    ((failures += 1))
  fi
}

run_expect_fail() {
  local rel="$1"
  local pattern="$2"
  local file="$cases_root/$rel"
  local arch
  local out="$work_dir/$(basename "$file").out"
  local cubin="$work_dir/$(basename "$file").cubin"

  ((ran += 1))
  arch="$(ptx_target "$file")"
  if [[ -z "$arch" ]]; then
    echo "[FAIL] missing .target in $file"
    ((failures += 1))
    return
  fi

  if ptxas -arch="$arch" "$file" -o "$cubin" >"$out" 2>&1; then
    echo "[FAIL] expected ptxas failure: $rel"
    ((failures += 1))
    return
  fi

  if [[ -n "$pattern" ]] && ! grep -Fq -- "$pattern" "$out"; then
    echo "[FAIL] expected ptxas error text not found: $rel"
    echo "[FAIL] missing pattern: $pattern"
    cat "$out"
    ((failures += 1))
    return
  fi

  echo "[PASS] ptxas rejected as expected: $rel"
}

ptxas_accepts_case() {
  local rel="$1"
  local file="$cases_root/$rel"
  local arch
  local out="$work_dir/probe_$(basename "$file").out"
  local cubin="$work_dir/probe_$(basename "$file").cubin"
  arch="$(ptx_target "$file")"
  if [[ -z "$arch" ]]; then
    return 1
  fi
  if ptxas -arch="$arch" "$file" -o "$cubin" >"$out" 2>&1; then
    return 0
  fi
  return 1
}

spec_tracking_ue8m0x2_bf16x2_relu="${LOWP_CVT_SPEC_TRACK_UE8M0X2_BF16X2_RELU:-auto}"
spec_tracking_case_rel="pass/instruction_cvt_ue8m0x2_bf16x2_relu_allowed.ptx"
pass_case_count=0
while IFS= read -r case_file; do
  rel="${case_file#$cases_root/}"
  if [[ "$rel" == "$spec_tracking_case_rel" ]]; then
    continue
  fi
  run_expect_pass "$rel"
  ((pass_case_count += 1))
done < <(
  find "$cases_root/pass" -maxdepth 1 -type f \
    -name "instruction_cvt_*_allowed.ptx" | sort
)
echo "[INFO] auto-covered parser-pass cvt forms: $pass_case_count"

case "$spec_tracking_ue8m0x2_bf16x2_relu" in
  auto)
    if ptxas_accepts_case "$spec_tracking_case_rel"; then
      echo "[SPEC-TRACKING:ue8m0x2_bf16x2_relu] ptxas currently accepts this PTX-legal form"
      run_expect_pass "$spec_tracking_case_rel"
    else
      echo "[SPEC-TRACKING:ue8m0x2_bf16x2_relu] ptxas currently rejects this PTX-legal form (tracked divergence)"
      run_expect_fail "$spec_tracking_case_rel" ""
    fi
    ;;
  ptxas_rejects)
    echo "[SPEC-TRACKING:ue8m0x2_bf16x2_relu] expecting ptxas rejection"
    run_expect_fail "$spec_tracking_case_rel" "Illegal modifier '.relu'"
    ;;
  ptxas_accepts)
    echo "[SPEC-TRACKING:ue8m0x2_bf16x2_relu] expecting ptxas acceptance"
    run_expect_pass "$spec_tracking_case_rel"
    ;;
  spec_accepts)
    echo "[SPEC-TRACKING:ue8m0x2_bf16x2_relu] enforcing spec expectation (accept)"
    run_expect_pass "$spec_tracking_case_rel"
    ;;
  *)
    echo "[FAIL] invalid LOWP_CVT_SPEC_TRACK_UE8M0X2_BF16X2_RELU value: $spec_tracking_ue8m0x2_bf16x2_relu (expected auto|ptxas_rejects|ptxas_accepts|spec_accepts)" >&2
    exit 2
    ;;
esac

run_expect_fail "fail/instruction_cvt_ue8m0x2_f32_relu_not_allowed.ptx" "Illegal modifier '.relu'"
run_expect_fail "fail/instruction_cvt_ue8m0x2_f32_sm120_not_allowed.ptx" "not supported on .target 'sm_120'"
run_expect_fail "fail/instruction_cvt_e2m1x2_f32_sm120_not_allowed.ptx" "not supported on .target 'sm_120'"
run_expect_fail "fail/instruction_cvt_f32_bf16_ftz_sm80_not_allowed.ptx" "requires .target sm_90 or higher"
run_expect_fail "fail/instruction_cvt_bf16_f16_sm80_not_allowed.ptx" "requires .target sm_90 or higher"
run_expect_fail "fail/instruction_cvt_f16_bf16_sm80_not_allowed.ptx" "requires .target sm_90 or higher"
run_expect_fail "fail/instruction_cvt_bf16_s32_sm80_not_allowed.ptx" "requires .target sm_90 or higher"
run_expect_fail "fail/instruction_cvt_s32_bf16_sm80_not_allowed.ptx" "requires .target sm_90 or higher"
run_expect_fail "fail/instruction_cvt_rs_sm110a_not_allowed.ptx" "Feature '.rs' not supported"
run_expect_fail "fail/instruction_cvt_rs_sm100_not_allowed.ptx" "Feature '.rs' not supported"
run_expect_fail "fail/instruction_cvt_ue8m0x2_bf16x2_missing_rounding_not_allowed.ptx" "Rounding modifier required for instruction 'cvt'"
run_expect_fail "fail/instruction_cvt_s2f6x2_bf16x2_missing_satfinite_not_allowed.ptx" "'.satfinite' modifier required"
run_expect_fail "fail/instruction_cvt_s2f6x2_bf16x2_scaled_without_operand_not_allowed.ptx" "Illegal modifier '.scaled::n2::ue8m0'"
run_expect_fail "fail/instruction_cvt_s2f6x2_bf16x2_operand_without_scaled_not_allowed.ptx" "Arguments mismatch for instruction 'cvt'"
run_expect_fail "fail/instruction_cvt_s2f6x2_bf16x2_sm100f_not_allowed.ptx" "not supported on .target 'sm_100f'"
run_expect_fail "fail/instruction_cvt_s2f6x2_bf16x2_sm120f_not_allowed.ptx" "not supported on .target 'sm_120f'"
run_expect_fail "fail/instruction_cvt_tf32_f32_satfinite_rn_sm90_not_allowed.ptx" "requires .target sm_100 or higher"
run_expect_fail "fail/instruction_cvt_tf32_f32_rna_relu_not_allowed.ptx" "cannot be combined with modifier '.rna'"
run_expect_fail "fail/instruction_cvt_ue4m3_f32_not_allowed.ptx" "Unexpected instruction types specified for 'cvt'"
run_expect_fail "fail/instruction_cvt_ue8m0x2_scaled_not_allowed.ptx" "Illegal modifier '.scaled::n2::ue8m0'"
run_expect_fail "fail/instruction_cvt_rs_sm100f_not_allowed.ptx" "Feature '.rs' not supported"
run_expect_fail "fail/instruction_cvt_rs_sm120a_not_allowed.ptx" "Feature '.rs' not supported"
run_expect_fail "fail/instruction_cvt_rs_sm120f_not_allowed.ptx" "Feature '.rs' not supported"
run_expect_fail "fail/instruction_cvt_s32_bf16_sat_not_allowed.ptx" "Illegal modifier '.sat'"
run_expect_fail "fail/instruction_cvt_s32_s32_sat_not_allowed.ptx" "Illegal modifier '.sat'"

if [[ $failures -ne 0 ]]; then
  echo "lowp cvt ptxas cross-check failed: $failures / $ran cases failed" >&2
  exit 1
fi

echo "lowp cvt ptxas cross-check passed: $ran cases"
