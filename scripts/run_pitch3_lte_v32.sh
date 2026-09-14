#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
DEVICE="${LTE_DEVICE:-cuda:1}"
STAGE="${1:-all}"
DATA_ROOT="${LTE_DATA_ROOT:-${PROJECT_ROOT}/runs/pitch3_lte_v3}"
RUN_ROOT="${LTE_V32_RUN_ROOT:-${PROJECT_ROOT}/runs/pitch3_lte_v32}"
SOURCE_MANIFEST="${LTE_SOURCE_MANIFEST:-${PROJECT_ROOT}/runs/pitch3_ltch/ood_v2/pitch3_training_manifest_augmented.csv}"
DATA_DIR="${DATA_ROOT}/exact_local_dataset"
MODEL_DIR="${RUN_ROOT}/models"
MODEL_SCREEN_DIR="${RUN_ROOT}/development_screen_model_only_fp32"
GUIDANCE_DIR="${RUN_ROOT}/development_guidance"
SCREEN_DIR="${RUN_ROOT}/development_screen"
QUALITY_REPORT="${GUIDANCE_DIR}/pitch3_lte_quality_report.json"

train_model() {
  "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/train_pitch3_lte.py" \
    --fingerprint "${PROJECT_ROOT}/metadata/focus_pitch3_fingerprint_v1.json" \
    --manifest "${DATA_DIR}/pitch3_lte_examples.csv" \
    --config "${PROJECT_ROOT}/configs/pitch3_lte_v32.toml" \
    --output-dir "${MODEL_DIR}" \
    --device "${DEVICE}"
}

checkpoint_path() {
  "${PYTHON_BIN}" -c \
    "import json; print(json.load(open('${MODEL_DIR}/pitch3_lte_manifest.json'))['checkpoint'])"
}

checkpoint_sha256() {
  "${PYTHON_BIN}" -c \
    "import json; print(json.load(open('${MODEL_DIR}/pitch3_lte_manifest.json'))['checkpoint_sha256'])"
}

model_screen() {
  local checkpoint checkpoint_hash
  checkpoint="$(checkpoint_path)"
  checkpoint_hash="$(checkpoint_sha256)"
  "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/evaluate_pitch3_lte.py" screen-development \
    --fingerprint "${PROJECT_ROOT}/metadata/focus_pitch3_fingerprint_v1.json" \
    --manifest "${DATA_DIR}/pitch3_lte_examples.csv" \
    --checkpoint "${checkpoint}" \
    --checkpoint-sha256 "${checkpoint_hash}" \
    --output-dir "${MODEL_SCREEN_DIR}" \
    --device "${DEVICE}"
}

require_model_screen() {
  "${PYTHON_BIN}" -c \
    "import json,sys; p=json.load(open('${MODEL_SCREEN_DIR}/pitch3_lte_development_screen.json')); ok=bool(p['metrics']['all_gates_passed']); print({'model_gates_passed': ok}); sys.exit(0 if ok else 2)"
}

development_guidance() {
  local checkpoint checkpoint_hash
  require_model_screen
  checkpoint="$(checkpoint_path)"
  checkpoint_hash="$(checkpoint_sha256)"
  "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/evaluate_pitch3_lte.py" materialize-guidance \
    --root "${PROJECT_ROOT}" \
    --fingerprint "${PROJECT_ROOT}/metadata/focus_pitch3_fingerprint_v1.json" \
    --manifest "${DATA_DIR}/pitch3_lte_examples.csv" \
    --source-manifest "${SOURCE_MANIFEST}" \
    --checkpoint "${checkpoint}" \
    --checkpoint-sha256 "${checkpoint_hash}" \
    --output-dir "${GUIDANCE_DIR}" \
    --device "${DEVICE}"
}

quality_evidence() {
  : "${LTE_CLAP_REVISION:?set LTE_CLAP_REVISION to the immutable 40-hex CLAP commit}"
  : "${LTE_QUALITY_TABLE:?set LTE_QUALITY_TABLE to the blind 256-pair quality CSV}"
  "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/build_ltsn_development_noninferiority_metrics.py" \
    --root "${PROJECT_ROOT}" \
    --run-root "${GUIDANCE_DIR}" \
    --model-revision "${LTE_CLAP_REVISION}" \
    --device "${DEVICE}" \
    --quality-table "${LTE_QUALITY_TABLE}" \
    --output "${GUIDANCE_DIR}/development_noninferiority_metrics.csv"
  "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/evaluate_pitch3_lte.py" finalize-quality \
    --metrics "${GUIDANCE_DIR}/development_noninferiority_metrics.csv" \
    --output "${QUALITY_REPORT}"
}

development_screen() {
  local checkpoint checkpoint_hash
  local quality_args=()
  checkpoint="$(checkpoint_path)"
  checkpoint_hash="$(checkpoint_sha256)"
  if [[ -f "${QUALITY_REPORT}" ]]; then
    quality_args=(--quality-report "${QUALITY_REPORT}")
  fi
  "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/evaluate_pitch3_lte.py" screen-development \
    --fingerprint "${PROJECT_ROOT}/metadata/focus_pitch3_fingerprint_v1.json" \
    --manifest "${DATA_DIR}/pitch3_lte_examples.csv" \
    --checkpoint "${checkpoint}" \
    --checkpoint-sha256 "${checkpoint_hash}" \
    --guidance-summary "${GUIDANCE_DIR}/pitch3_lte_guidance_summary.json" \
    --output-dir "${SCREEN_DIR}" \
    --device "${DEVICE}" \
    "${quality_args[@]}"
}

case "${STAGE}" in
  train) train_model ;;
  model-screen) model_screen ;;
  guidance) development_guidance ;;
  quality) quality_evidence ;;
  screen) development_screen ;;
  all)
    train_model
    model_screen
    development_guidance
    development_screen
    ;;
  *)
    echo "usage: $0 {train|model-screen|guidance|quality|screen|all}" >&2
    exit 2
    ;;
esac
