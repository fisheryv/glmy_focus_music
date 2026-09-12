#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/ACE-Step-1.5/.venv/bin/python}"
PIPELINE_ROOT="${PIPELINE_ROOT:-runs/topology_rerank_lora_v1/lora_v2}"
LORA_CONFIG="${LORA_CONFIG:-configs/topology_lora_v2.json}"
TEACHER_CONFIG="${TEACHER_CONFIG:-configs/topology_lora_teacher_v2.toml}"
SELECTOR_CONFIG="${SELECTOR_CONFIG:-configs/constrained_reranker_v2.json}"
RERANK_GATE="${RERANK_GATE:-${PROJECT_ROOT}/metadata/ace_constrained_reranking_effect_gate_v2.json}"
FROZEN_SELECTOR="${FROZEN_SELECTOR:-${PROJECT_ROOT}/runs/topology_rerank_lora_v1/constrained_v2/frozen_selector.json}"
STAGE="${1:-}"

export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}/packages/pyglmy/src${PYTHONPATH:+:${PYTHONPATH}}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python environment not found: ${PYTHON_BIN}" >&2
  exit 2
fi

PROMPT_DIR="${PROJECT_ROOT}/${PIPELINE_ROOT}/prompts"
TEACHER_RUN="${PROJECT_ROOT}/${PIPELINE_ROOT}/rerank/topology_lora_teacher_constrained_v2"
TEACHER_SELECTION="${TEACHER_RUN}/constrained"
TEACHER_DIR="${PROJECT_ROOT}/${PIPELINE_ROOT}/teacher"
TENSOR_DIR="${PROJECT_ROOT}/${PIPELINE_ROOT}/tensors"
LORA_OUTPUT="${PROJECT_ROOT}/${PIPELINE_ROOT}/lora"
LORA_ARTIFACT="${PROJECT_ROOT}/${PIPELINE_ROOT}/lora_artifact.json"
VALIDATION_ROOT="${PROJECT_ROOT}/${PIPELINE_ROOT}/validation"
SCALE_SELECTION="${PROJECT_ROOT}/${PIPELINE_ROOT}/scale_selection.json"

LORA_COMMON=(--root "${PROJECT_ROOT}" --config "${LORA_CONFIG}")

require_v2_authority() {
  if [[ ! -f "${RERANK_GATE}" ]]; then
    echo "Passed constrained-reranker v2 gate is required: ${RERANK_GATE}" >&2
    exit 2
  fi
  if [[ ! -f "${FROZEN_SELECTOR}" ]]; then
    echo "Frozen constrained selector is required: ${FROZEN_SELECTOR}" >&2
    exit 2
  fi
  "${PYTHON_BIN}" -m generation.topology_lora_cli check-gate \
    "${LORA_COMMON[@]}" --reranking-gate "${RERANK_GATE}"
}

case "${STAGE}" in
  prepare-prompts)
    "${PYTHON_BIN}" -m generation.topology_lora_cli prepare-prompts \
      "${LORA_COMMON[@]}" --output-dir "${PROMPT_DIR}"
    ;;
  check-gate)
    require_v2_authority
    ;;
  teacher-run)
    require_v2_authority
    read -r -a TEACHER_DEVICE_ARGS <<< "${TEACHER_DEVICES:-cuda:0 cuda:1 cuda:2 cuda:3}"
    "${PYTHON_BIN}" -m generation.rerank_cli preflight \
      --root "${PROJECT_ROOT}" --config "${TEACHER_CONFIG}" --backend ace \
      --devices "${TEACHER_DEVICE_ARGS[@]}"
    "${PYTHON_BIN}" -m generation.rerank_cli run \
      --root "${PROJECT_ROOT}" --config "${TEACHER_CONFIG}" \
      --backend ace --devices "${TEACHER_DEVICE_ARGS[@]}" --retry-failed
    ;;
  teacher-semantics)
    require_v2_authority
    "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/run_constrained_reranker.py" build-semantics \
      --root "${PROJECT_ROOT}" --run-config "${TEACHER_CONFIG}" \
      --selector-config "${SELECTOR_CONFIG}" --device "${CLAP_DEVICE:-cuda:0}"
    ;;
  teacher-apply)
    require_v2_authority
    "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/run_constrained_reranker.py" apply \
      --root "${PROJECT_ROOT}" --run-config "${TEACHER_CONFIG}" \
      --selector-config "${SELECTOR_CONFIG}" --frozen-selector "${FROZEN_SELECTOR}"
    ;;
  export-teacher)
    require_v2_authority
    "${PYTHON_BIN}" -m generation.topology_lora_cli export-teacher \
      "${LORA_COMMON[@]}" --reranking-run "${TEACHER_RUN}" \
      --prompt-manifest "${PROMPT_DIR}/train.csv" \
      --reranking-gate "${RERANK_GATE}" \
      --selection-contract "${TEACHER_SELECTION}/selection_contract.json" \
      --output-dir "${TEACHER_DIR}"
    ;;
  preprocess-lora)
    "${PYTHON_BIN}" -m generation.topology_lora_cli preprocess \
      "${LORA_COMMON[@]}" --teacher-dir "${TEACHER_DIR}" \
      --tensor-dir "${TENSOR_DIR}" --lora-output "${LORA_OUTPUT}" \
      --python-bin "${PYTHON_BIN}"
    ;;
  train-lora)
    "${PYTHON_BIN}" -m generation.topology_lora_cli train \
      "${LORA_COMMON[@]}" --teacher-dir "${TEACHER_DIR}" \
      --tensor-dir "${TENSOR_DIR}" --lora-output "${LORA_OUTPUT}" \
      --python-bin "${PYTHON_BIN}"
    ;;
  finalize-lora)
    "${PYTHON_BIN}" -m generation.topology_lora_cli finalize \
      "${LORA_COMMON[@]}" --teacher-dir "${TEACHER_DIR}" \
      --lora-output "${LORA_OUTPUT}"
    ;;
  validate-development)
    if [[ -z "${LORA_SCALE:-}" ]]; then
      echo "Set LORA_SCALE to one frozen development scale: 0.5, 0.75, or 1.0." >&2
      exit 2
    fi
    "${PYTHON_BIN}" -m generation.topology_lora_cli validate \
      "${LORA_COMMON[@]}" --rerank-config "${TEACHER_CONFIG}" \
      --lora-artifact "${LORA_ARTIFACT}" \
      --prompt-manifest "${PROMPT_DIR}/development.csv" --split development \
      --scale "${LORA_SCALE}" --seed-start 2026140100 \
      --validation-run "${VALIDATION_ROOT}/development_scale_${LORA_SCALE}"
    ;;
  select-scale)
    "${PYTHON_BIN}" -m generation.topology_lora_cli select-scale \
      "${LORA_COMMON[@]}" --validation-root "${VALIDATION_ROOT}" \
      --scale-selection "${SCALE_SELECTION}"
    ;;
  validate-qualification)
    if [[ -z "${LORA_SCALE:-}" ]]; then
      echo "Set LORA_SCALE to the single scale selected on development." >&2
      exit 2
    fi
    "${PYTHON_BIN}" -m generation.topology_lora_cli validate \
      "${LORA_COMMON[@]}" --rerank-config "${TEACHER_CONFIG}" \
      --lora-artifact "${LORA_ARTIFACT}" \
      --prompt-manifest "${PROMPT_DIR}/qualification.csv" --split qualification \
      --scale "${LORA_SCALE}" --scale-selection "${SCALE_SELECTION}" \
      --seed-start 2026150100 \
      --validation-run "${VALIDATION_ROOT}/qualification"
    ;;
  *)
    echo "Usage: $0 {prepare-prompts|check-gate|teacher-run|teacher-semantics|teacher-apply|export-teacher|preprocess-lora|train-lora|finalize-lora|validate-development|select-scale|validate-qualification}" >&2
    exit 2
    ;;
esac
