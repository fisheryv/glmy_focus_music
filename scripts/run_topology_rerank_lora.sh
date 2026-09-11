#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/ACE-Step-1.5/.venv/bin/python}"
RERANK_CONFIG="${RERANK_CONFIG:-configs/topology_rerank_180s.toml}"
LORA_CONFIG="${LORA_CONFIG:-configs/topology_lora_v1.json}"
PIPELINE_ROOT="${PIPELINE_ROOT:-runs/topology_rerank_lora_v1}"
FORMAL_RUN_ID="${FORMAL_RUN_ID:-topology_bestof8_formal_v1}"
TEACHER_RUN_ID="${TEACHER_RUN_ID:-topology_lora_teacher_bestof8_v1}"
RERANK_GATE="${RERANK_GATE:-${PROJECT_ROOT}/metadata/ace_reranking_effect_gate.json}"
STAGE="${1:-}"

export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}/packages/pyglmy/src${PYTHONPATH:+:${PYTHONPATH}}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python environment not found: ${PYTHON_BIN}" >&2
  exit 2
fi

RERANK_COMMON=(--root "${PROJECT_ROOT}" --config "${RERANK_CONFIG}" --backend ace)
LORA_COMMON=(--root "${PROJECT_ROOT}" --config "${LORA_CONFIG}")
PROMPT_DIR="${PIPELINE_ROOT}/prompts"
TEACHER_DIR="${PIPELINE_ROOT}/teacher"
TENSOR_DIR="${PIPELINE_ROOT}/tensors"
LORA_OUTPUT="${PIPELINE_ROOT}/lora"

case "${STAGE}" in
  prepare-prompts)
    "${PYTHON_BIN}" -m generation.topology_lora_cli prepare-prompts \
      "${LORA_COMMON[@]}" --output-dir "${PROMPT_DIR}"
    ;;
  rerank-formal)
    "${PYTHON_BIN}" -m generation.rerank_cli preflight \
      "${RERANK_COMMON[@]}" --run-id "${FORMAL_RUN_ID}"
    "${PYTHON_BIN}" -m generation.rerank_cli run \
      "${RERANK_COMMON[@]}" --run-id "${FORMAL_RUN_ID}" --retry-failed
    ;;
  init-formal-evidence)
    "${PYTHON_BIN}" -m generation.rerank_cli init-evidence \
      "${RERANK_COMMON[@]}" --run-id "${FORMAL_RUN_ID}"
    ;;
  evaluate-formal-evidence)
    if [[ -z "${NONINFERIORITY_TABLE:-}" ]]; then
      echo "Set NONINFERIORITY_TABLE to the completed blind-evaluation CSV." >&2
      exit 2
    fi
    "${PYTHON_BIN}" -m generation.rerank_cli evaluate-evidence \
      "${RERANK_COMMON[@]}" --run-id "${FORMAL_RUN_ID}" \
      --noninferiority-table "${NONINFERIORITY_TABLE}"
    ;;
  issue-formal-gate)
    "${PYTHON_BIN}" -m generation.rerank_cli issue-gate \
      "${RERANK_COMMON[@]}" --run-id "${FORMAL_RUN_ID}" \
      --gate-output "${RERANK_GATE}"
    ;;
  rerank-train)
    if [[ ! -f "${RERANK_GATE}" ]]; then
      echo "Passed formal reranking gate is required before teacher generation." >&2
      exit 2
    fi
    "${PYTHON_BIN}" -m generation.topology_lora_cli check-gate \
      "${LORA_COMMON[@]}" --reranking-gate "${RERANK_GATE}"
    "${PYTHON_BIN}" -m generation.rerank_cli run \
      "${RERANK_COMMON[@]}" --run-id "${TEACHER_RUN_ID}" \
      --prompt-manifest "${PROMPT_DIR}/train.csv" --retry-failed
    ;;
  export-teacher)
    "${PYTHON_BIN}" -m generation.topology_lora_cli export-teacher \
      "${LORA_COMMON[@]}" \
      --reranking-run "${PIPELINE_ROOT}/rerank/${TEACHER_RUN_ID}" \
      --prompt-manifest "${PROMPT_DIR}/train.csv" \
      --reranking-gate "${RERANK_GATE}" --output-dir "${TEACHER_DIR}"
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
      "${LORA_COMMON[@]}" --rerank-config "${RERANK_CONFIG}" \
      --lora-artifact "${PIPELINE_ROOT}/lora_artifact.json" \
      --prompt-manifest "${PROMPT_DIR}/development.csv" --split development \
      --scale "${LORA_SCALE}" \
      --validation-run "${PIPELINE_ROOT}/validation/development_scale_${LORA_SCALE}"
    ;;
  select-scale)
    "${PYTHON_BIN}" -m generation.topology_lora_cli select-scale \
      "${LORA_COMMON[@]}" --validation-root "${PIPELINE_ROOT}/validation" \
      --scale-selection "${PIPELINE_ROOT}/scale_selection.json"
    ;;
  validate-qualification)
    if [[ -z "${LORA_SCALE:-}" ]]; then
      echo "Set LORA_SCALE to the single scale selected on development." >&2
      exit 2
    fi
    "${PYTHON_BIN}" -m generation.topology_lora_cli validate \
      "${LORA_COMMON[@]}" --rerank-config "${RERANK_CONFIG}" \
      --lora-artifact "${PIPELINE_ROOT}/lora_artifact.json" \
      --prompt-manifest "${PROMPT_DIR}/qualification.csv" --split qualification \
      --scale "${LORA_SCALE}" \
      --scale-selection "${PIPELINE_ROOT}/scale_selection.json" \
      --seed-start 2026092100 \
      --validation-run "${PIPELINE_ROOT}/validation/qualification"
    ;;
  *)
    echo "Usage: $0 {prepare-prompts|rerank-formal|init-formal-evidence|evaluate-formal-evidence|issue-formal-gate|rerank-train|export-teacher|preprocess-lora|train-lora|finalize-lora|validate-development|select-scale|validate-qualification}" >&2
    exit 2
    ;;
esac
