#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export LTE_TRAIN_CONFIG="${LTE_TRAIN_CONFIG:-${PROJECT_ROOT}/configs/pitch3_lte_v38a.toml}"
export LTE_V33_RUN_ROOT="${LTE_V38A_RUN_ROOT:-${PROJECT_ROOT}/runs/pitch3_lte_v38a}"
export LTE_V33_SEEDS="${LTE_V38A_SEEDS:-20260941 20260942 20260943}"
export LTE_VERSION_LABEL="V3.8-A"

if [[ "${1:-cv}" == "cv" ]]; then
  PYTHON_BIN="${PYTHON_BIN:-python}"
  export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
  DATA_ROOT="${LTE_DATA_ROOT:-${PROJECT_ROOT}/runs/pitch3_lte_v3}"
  CV_ROOT="${LTE_V38A_CV_ROOT:-${LTE_V33_RUN_ROOT}/cv}"
  DEVICES=(${LTE_DEVICES:-cuda:1 cuda:2 cuda:3})
  FOLDS=(${LTE_CV_FOLDS:-0 1 2 3 4})
  CV_ARGS=()
  if [[ "${LTE_CV_GLOBAL_ONLY:-1}" == "1" ]]; then
    CV_ARGS+=(--global-only)
  fi
  mkdir -p "${CV_ROOT}"
  pids=()
  failed=0
  for index in "${!FOLDS[@]}"; do
    fold="${FOLDS[$index]}"
    slot=$((index % ${#DEVICES[@]}))
    if [[ "${slot}" == "0" && "${#pids[@]}" -gt "0" ]]; then
      for pid in "${pids[@]}"; do wait "${pid}" || failed=1; done
      [[ "${failed}" == "0" ]] || exit 1
      pids=()
    fi
    "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/train_pitch3_lte.py" \
      --fingerprint "${LTE_FINGERPRINT:-${PROJECT_ROOT}/metadata/focus_pitch3_fingerprint_v1.json}" \
      --manifest "${DATA_ROOT}/exact_local_dataset/pitch3_lte_examples.csv" \
      --config "${LTE_TRAIN_CONFIG}" --seed "${LTE_CV_SEED:-20260941}" \
      --cv-fold "${fold}" "${CV_ARGS[@]}" \
      --output-dir "${CV_ROOT}/fold_${fold}/models" --device "${DEVICES[$slot]}" \
      >"${CV_ROOT}/fold_${fold}_train.log" 2>&1 &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "${pid}" || failed=1; done
  [[ "${failed}" == "0" ]] || exit 1
  "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/summarize_pitch3_lte_cv.py" --cv-root "${CV_ROOT}"
  exit 0
fi

exec bash "${PROJECT_ROOT}/scripts/run_pitch3_lte_v33.sh" "$@"
