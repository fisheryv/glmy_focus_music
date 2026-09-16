#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
STAGE="${1:-all}"
DATA_ROOT="${LTE_DATA_ROOT:-${PROJECT_ROOT}/runs/pitch3_lte_v3}"
RUN_ROOT="${LTE_V33_RUN_ROOT:-${PROJECT_ROOT}/runs/pitch3_lte_v33}"
CONFIG="${LTE_TRAIN_CONFIG:-${PROJECT_ROOT}/configs/pitch3_lte_v33.toml}"
FINGERPRINT="${LTE_FINGERPRINT:-${PROJECT_ROOT}/metadata/focus_pitch3_fingerprint_v1.json}"
VERSION_LABEL="${LTE_VERSION_LABEL:-V3.3}"
SOURCE_MANIFEST="${LTE_SOURCE_MANIFEST:-${PROJECT_ROOT}/runs/pitch3_ltch/ood_v2/pitch3_training_manifest_augmented.csv}"
DATA_DIR="${DATA_ROOT}/exact_local_dataset"
ENSEMBLE_MANIFEST="${RUN_ROOT}/pitch3_lte_ensemble.json"
MODEL_SCREEN_DIR="${RUN_ROOT}/development_screen_model_only_fp32"
GUIDANCE_DIR="${RUN_ROOT}/development_guidance"
FINAL_SCREEN_DIR="${RUN_ROOT}/development_screen"
QUALITY_REPORT="${GUIDANCE_DIR}/pitch3_lte_quality_report.json"
SEEDS=(${LTE_V33_SEEDS:-20260923 20260924 20260925})
DEVICES=(${LTE_DEVICES:-cuda:1 cuda:2 cuda:3})
mkdir -p "${RUN_ROOT}"

if [[ "${#SEEDS[@]}" -ne "${#DEVICES[@]}" ]]; then
  echo "seed list and LTE_DEVICES must contain the same number of entries" >&2
  exit 2
fi

seed_root() {
  echo "${RUN_ROOT}/seed_$1"
}

train_all() {
  local pids=()
  local training_args=()
  if [[ "${LTE_GLOBAL_ONLY:-0}" == "1" ]]; then
    training_args+=(--global-only)
  fi
  local index seed device root
  for index in "${!SEEDS[@]}"; do
    seed="${SEEDS[$index]}"
    device="${DEVICES[$index]}"
    root="$(seed_root "${seed}")"
    "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/train_pitch3_lte.py" \
      --fingerprint "${FINGERPRINT}" "${training_args[@]}" \
      --manifest "${DATA_DIR}/pitch3_lte_examples.csv" \
      --config "${CONFIG}" \
      --seed "${seed}" \
      --output-dir "${root}/models" \
      --device "${device}" \
      >"${root}_train.log" 2>&1 &
    pids+=("$!")
  done
  local failed=0
  for index in "${!pids[@]}"; do
    if ! wait "${pids[$index]}"; then
      echo "${VERSION_LABEL} seed ${SEEDS[$index]} failed; inspect ${RUN_ROOT}/seed_${SEEDS[$index]}_train.log" >&2
      failed=1
    fi
  done
  [[ "${failed}" -eq 0 ]]
}

build_ensemble() {
  local manifests=()
  local seed
  for seed in "${SEEDS[@]}"; do
    manifests+=(--manifest "$(seed_root "${seed}")/models/pitch3_lte_manifest.json")
  done
  "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/build_pitch3_lte_ensemble.py" \
    "${manifests[@]}" \
    --output "${ENSEMBLE_MANIFEST}"
}

screen_seeds() {
  local pids=()
  local index seed device root checkpoint checkpoint_hash
  for index in "${!SEEDS[@]}"; do
    seed="${SEEDS[$index]}"
    device="${DEVICES[$index]}"
    root="$(seed_root "${seed}")"
    checkpoint="${root}/models/pitch3_lte_seed_${seed}.pt"
    checkpoint_hash="$("${PYTHON_BIN}" -c "import json; print(json.load(open('${root}/models/pitch3_lte_manifest.json'))['checkpoint_sha256'])")"
    "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/evaluate_pitch3_lte.py" screen-development \
      --fingerprint "${FINGERPRINT}" \
      --manifest "${DATA_DIR}/pitch3_lte_examples.csv" \
      --checkpoint "${checkpoint}" \
      --checkpoint-sha256 "${checkpoint_hash}" \
      --output-dir "${root}/development_screen_model_only_fp32" \
      --device "${device}" \
      >"${root}_screen.log" 2>&1 &
    pids+=("$!")
  done
  local failed=0
  for index in "${!pids[@]}"; do
    if ! wait "${pids[$index]}"; then
      echo "${VERSION_LABEL} seed ${SEEDS[$index]} screen failed; inspect ${RUN_ROOT}/seed_${SEEDS[$index]}_screen.log" >&2
      failed=1
    fi
  done
  [[ "${failed}" -eq 0 ]]
}

screen_ensemble_model_only() {
  "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/evaluate_pitch3_lte.py" screen-development \
    --fingerprint "${FINGERPRINT}" \
    --manifest "${DATA_DIR}/pitch3_lte_examples.csv" \
    --ensemble-manifest "${ENSEMBLE_MANIFEST}" \
    --output-dir "${MODEL_SCREEN_DIR}" \
    --device "${DEVICES[0]}"
}

require_model_screen() {
  "${PYTHON_BIN}" -c \
    "import json,sys; p=json.load(open('${MODEL_SCREEN_DIR}/pitch3_lte_development_screen.json')); ok=bool(p['metrics']['all_gates_passed']); print({'model_gates_passed': ok}); sys.exit(0 if ok else 2)"
}

development_guidance() {
  require_model_screen
  "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/evaluate_pitch3_lte.py" materialize-guidance \
    --root "${PROJECT_ROOT}" \
    --fingerprint "${FINGERPRINT}" \
    --manifest "${DATA_DIR}/pitch3_lte_examples.csv" \
    --source-manifest "${SOURCE_MANIFEST}" \
    --ensemble-manifest "${ENSEMBLE_MANIFEST}" \
    --output-dir "${GUIDANCE_DIR}" \
    --device "${DEVICES[0]}"
}

quality_evidence() {
  : "${LTE_CLAP_REVISION:?set LTE_CLAP_REVISION to the immutable 40-hex CLAP commit}"
  : "${LTE_QUALITY_TABLE:?set LTE_QUALITY_TABLE to the blind 256-pair quality CSV}"
  "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/build_ltsn_development_noninferiority_metrics.py" \
    --root "${PROJECT_ROOT}" \
    --run-root "${GUIDANCE_DIR}" \
    --model-revision "${LTE_CLAP_REVISION}" \
    --device "${DEVICES[0]}" \
    --quality-table "${LTE_QUALITY_TABLE}" \
    --output "${GUIDANCE_DIR}/development_noninferiority_metrics.csv"
  "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/evaluate_pitch3_lte.py" finalize-quality \
    --metrics "${GUIDANCE_DIR}/development_noninferiority_metrics.csv" \
    --output "${QUALITY_REPORT}"
}

final_screen() {
  local quality_args=()
  if [[ -f "${QUALITY_REPORT}" ]]; then
    quality_args=(--quality-report "${QUALITY_REPORT}")
  fi
  "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/evaluate_pitch3_lte.py" screen-development \
    --fingerprint "${FINGERPRINT}" \
    --manifest "${DATA_DIR}/pitch3_lte_examples.csv" \
    --ensemble-manifest "${ENSEMBLE_MANIFEST}" \
    --guidance-summary "${GUIDANCE_DIR}/pitch3_lte_guidance_summary.json" \
    --output-dir "${FINAL_SCREEN_DIR}" \
    --device "${DEVICES[0]}" \
    "${quality_args[@]}"
}

case "${STAGE}" in
  train) train_all ;;
  ensemble) build_ensemble ;;
  model-screen)
    screen_seeds
    screen_ensemble_model_only
    ;;
  guidance) development_guidance ;;
  quality) quality_evidence ;;
  screen) final_screen ;;
  all)
    train_all
    build_ensemble
    screen_seeds
    screen_ensemble_model_only
    ;;
  *)
    echo "usage: $0 {train|ensemble|model-screen|guidance|quality|screen|all}" >&2
    exit 2
    ;;
esac
