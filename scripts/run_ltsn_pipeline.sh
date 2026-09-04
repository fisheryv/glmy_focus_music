#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/ACE-Step-1.5/.venv/bin/python}"
STAGE="${1:-}"
RUN_ROOT="${RUN_ROOT:-${PROJECT_ROOT}/runs/ltsn_turbo}"
PROMPT_MANIFEST="${PROMPT_MANIFEST:-${PROJECT_ROOT}/metadata/ltsn_prompts.csv}"
SURROGATE_TRAINING_GATE="${SURROGATE_TRAINING_GATE:-${PROJECT_ROOT}/metadata/ltsn_surrogate_training_gate.json}"
FINGERPRINT="${PROJECT_ROOT}/metadata/focus_path_homology_fingerprint_v2.json"
CONFIG="${PROJECT_ROOT}/configs/ltsn_training.toml"

[[ -x "${PYTHON_BIN}" ]] || { echo "Python environment is missing: ${PYTHON_BIN}" >&2; exit 2; }

collect() {
  : "${ACE_MODEL_SHA256:?Set ACE_MODEL_SHA256 to the 64-hex model tree digest}"
  : "${VAE_SHA256:?Set VAE_SHA256 to the 64-hex VAE tree digest}"
  "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/collect_ltsn_trajectories.py" \
    --root "${PROJECT_ROOT}" \
    --ace-config "${PROJECT_ROOT}/configs/ace_rerank_180s.toml" \
    --prompt-manifest "${PROMPT_MANIFEST}" \
    --output-dir "${RUN_ROOT}/trajectories" \
    --backend ace \
    --ace-model-sha256 "${ACE_MODEL_SHA256}" \
    --vae-sha256 "${VAE_SHA256}" \
    --seeds-per-prompt "${LTSN_SEEDS_PER_PROMPT:-4}" \
    --duration-seconds 180 \
    --decode-snapshots \
    --discard-generator-final-audio
}

labels() {
  [[ -f "${SURROGATE_TRAINING_GATE}" ]] || { echo "Passed ltsn_surrogate_training_v1 gate is required: ${SURROGATE_TRAINING_GATE}" >&2; exit 3; }
  "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/build_ltsn_labels.py" \
    --root "${PROJECT_ROOT}" \
    --fingerprint "${FINGERPRINT}" \
    --trajectory-manifest "${RUN_ROOT}/trajectories/trajectory_manifest.csv" \
    --work-dir "${RUN_ROOT}/exact_work" \
    --output-dir "${RUN_ROOT}/labels" \
    --surrogate-training-gate "${SURROGATE_TRAINING_GATE}" \
    --workers "${EXACT_WORKERS:-16}" \
    --batch-size "${EXACT_BATCH_SIZE:-256}" \
    --materialize-mode "${MATERIALIZE_MODE:-auto}"
}

train() {
  [[ -f "${SURROGATE_TRAINING_GATE}" ]] || { echo "Passed ltsn_surrogate_training_v1 gate is required: ${SURROGATE_TRAINING_GATE}" >&2; exit 3; }
  local -a train_device_args
  if [[ -n "${TRAIN_DEVICES:-}" ]]; then
    local -a train_devices
    IFS=',' read -r -a train_devices <<< "${TRAIN_DEVICES}"
    train_device_args=(--devices "${train_devices[@]}")
  else
    train_device_args=(--device "${TRAIN_DEVICE:-cuda:0}")
  fi
  "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/train_path_homology_surrogate.py" \
    --fingerprint "${FINGERPRINT}" \
    --manifest "${RUN_ROOT}/labels/ltsn_manifest.csv" \
    --split-manifest "${RUN_ROOT}/labels/split_manifest.json" \
    --config "${CONFIG}" \
    --output-dir "${RUN_ROOT}/models" \
    --surrogate-training-gate "${SURROGATE_TRAINING_GATE}" \
    "${train_device_args[@]}"
}

calibrate() {
  "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/evaluate_ltsn_qualification.py" calibrate \
    --fingerprint "${FINGERPRINT}" \
    --manifest "${RUN_ROOT}/labels/ltsn_manifest.csv" \
    --ensemble-manifest "${RUN_ROOT}/models/ensemble_manifest.json" \
    --output "${RUN_ROOT}/calibration.json" \
    --device "${EVAL_DEVICE:-cuda:0}"
}

qualify() {
  [[ -f "${RUN_ROOT}/guidance_development.json" ]] || {
    echo "Run guidance-development with a decoded exact pair table before qualification" >&2
    exit 3
  }
  "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/evaluate_ltsn_qualification.py" qualify \
    --fingerprint "${FINGERPRINT}" \
    --manifest "${RUN_ROOT}/labels/ltsn_manifest.csv" \
    --ensemble-manifest "${RUN_ROOT}/models/ensemble_manifest.json" \
    --calibration "${RUN_ROOT}/calibration.json" \
    --guidance-development-report "${RUN_ROOT}/guidance_development.json" \
    --output "${RUN_ROOT}/qualification.json" \
    --device "${EVAL_DEVICE:-cuda:0}"
}

guidance_development() {
  : "${PAIR_TABLE:?Set PAIR_TABLE to the decoded exact/proxy development pair CSV}"
  "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/evaluate_path_homology_guidance.py" \
    --fingerprint "${FINGERPRINT}" \
    --pair-table "${PAIR_TABLE}" \
    --output "${RUN_ROOT}/guidance_development.json" \
    --mode development
}

guidance_confirmation() {
  : "${PAIR_TABLE:?Set PAIR_TABLE to the fresh 32-prompt x 8-seed confirmation pair CSV}"
  [[ -f "${RUN_ROOT}/qualification.json" ]] || {
    echo "Passed independent qualification report is required: ${RUN_ROOT}/qualification.json" >&2
    exit 3
  }
  "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/evaluate_path_homology_guidance.py" \
    --fingerprint "${FINGERPRINT}" \
    --pair-table "${PAIR_TABLE}" \
    --output "${RUN_ROOT}/guidance_confirmation.json" \
    --qualification-report "${RUN_ROOT}/qualification.json" \
    --mode confirmation
}

case "${STAGE}" in
  collect) collect ;;
  labels) labels ;;
  train) train ;;
  calibrate) calibrate ;;
  guidance-development) guidance_development ;;
  qualify) qualify ;;
  guidance-confirmation) guidance_confirmation ;;
  *) echo "Usage: $0 {collect|labels|train|calibrate|guidance-development|qualify|guidance-confirmation}" >&2; exit 2 ;;
esac
