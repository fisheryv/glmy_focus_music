#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
DEVICES="${LTE_DEVICES:-}"
DEVICE="${LTE_DEVICE:-}"
if [[ -z "${DEVICE}" ]]; then
  DEVICE="${DEVICES%% *}"
  DEVICE="${DEVICE:-cuda:1}"
fi
WORKERS_PER_DEVICE="${LTE_WORKERS_PER_DEVICE:-4}"
EXACT_BATCH_SIZE="${LTE_EXACT_BATCH_SIZE:-32}"
STAGE="${1:-all}"
RUN_ROOT="${LTE_RUN_ROOT:-${PROJECT_ROOT}/runs/pitch3_lte_v3}"
SOURCE_MANIFEST="${LTE_SOURCE_MANIFEST:-${PROJECT_ROOT}/runs/pitch3_ltch/ood_v2/pitch3_training_manifest_augmented.csv}"
ACE_MODEL_SHA256="${LTE_ACE_MODEL_SHA256:-518e5c28530e8db0bfb275c327ca7a5b920b36e2345c3b4178d4d51fccdc6a7c}"
PROMPT_DIR="${RUN_ROOT}/prompt_embeddings"
DATA_DIR="${RUN_ROOT}/exact_local_dataset"
MODEL_DIR="${RUN_ROOT}/models"
GUIDANCE_DIR="${RUN_ROOT}/development_guidance"
SCREEN_DIR="${RUN_ROOT}/development_screen"
QUALITY_REPORT="${GUIDANCE_DIR}/pitch3_lte_quality_report.json"

prompt_embeddings() {
  "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/build_pitch3_lte_prompt_embeddings.py" \
    --root "${PROJECT_ROOT}" \
    --source-manifest "${SOURCE_MANIFEST}" \
    --output-dir "${PROMPT_DIR}" \
    --ace-model-sha256 "${ACE_MODEL_SHA256}" \
    --device "${DEVICE}"
}

local_dataset() {
  local -a device_args device_list
  if [[ -n "${DEVICES}" ]]; then
    read -r -a device_list <<< "${DEVICES}"
    if (( ${#device_list[@]} < 2 )); then
      echo "LTE_DEVICES must contain at least two devices" >&2
      exit 2
    fi
    device_args=(--devices "${device_list[@]}" --workers-per-device "${WORKERS_PER_DEVICE}")
  else
    device_args=(--device "${DEVICE}")
  fi
  "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/build_pitch3_lte_local_pairs.py" \
    --root "${PROJECT_ROOT}" \
    --source-manifest "${SOURCE_MANIFEST}" \
    --prompt-embeddings "${PROMPT_DIR}/pitch3_lte_prompt_embeddings.csv" \
    --output-dir "${DATA_DIR}" \
    --include-splits train development \
    --local-splits train development \
    --exact-batch-size "${EXACT_BATCH_SIZE}" \
    "${device_args[@]}"
}

train_model() {
  "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/train_pitch3_lte.py" \
    --manifest "${DATA_DIR}/pitch3_lte_examples.csv" \
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

development_guidance() {
  local checkpoint
  local checkpoint_hash
  checkpoint="$(checkpoint_path)"
  checkpoint_hash="$(checkpoint_sha256)"
  "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/evaluate_pitch3_lte.py" materialize-guidance \
    --root "${PROJECT_ROOT}" \
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
  local checkpoint
  local checkpoint_hash
  local quality_args=()
  checkpoint="$(checkpoint_path)"
  checkpoint_hash="$(checkpoint_sha256)"
  if [[ -f "${QUALITY_REPORT}" ]]; then
    quality_args=(--quality-report "${QUALITY_REPORT}")
  fi
  "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/evaluate_pitch3_lte.py" screen-development \
    --manifest "${DATA_DIR}/pitch3_lte_examples.csv" \
    --checkpoint "${checkpoint}" \
    --checkpoint-sha256 "${checkpoint_hash}" \
    --guidance-summary "${GUIDANCE_DIR}/pitch3_lte_guidance_summary.json" \
    --output-dir "${SCREEN_DIR}" \
    --device "${DEVICE}" \
    "${quality_args[@]}"
}

case "${STAGE}" in
  prompt) prompt_embeddings ;;
  data) local_dataset ;;
  train) train_model ;;
  guidance) development_guidance ;;
  quality) quality_evidence ;;
  screen) development_screen ;;
  all)
    prompt_embeddings
    local_dataset
    train_model
    development_guidance
    development_screen
    ;;
  *)
    echo "usage: $0 {prompt|data|train|guidance|quality|screen|all}" >&2
    exit 2
    ;;
esac
