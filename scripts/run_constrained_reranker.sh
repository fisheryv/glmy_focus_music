#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/ACE-Step-1.5/.venv/bin/python}"
PIPELINE_ROOT="${PIPELINE_ROOT:-runs/topology_rerank_lora_v1/constrained}"
SELECTOR_CONFIG="${SELECTOR_CONFIG:-configs/constrained_reranker_v1.json}"
CALIBRATION_CONFIG="${CALIBRATION_CONFIG:-configs/topology_constrained_calibration.toml}"
CONFIRMATION_CONFIG="${CONFIRMATION_CONFIG:-configs/topology_constrained_confirmation.toml}"
FROZEN_SELECTOR="${PROJECT_ROOT}/${PIPELINE_ROOT}/frozen_selector.json"
GATE_OUTPUT="${GATE_OUTPUT:-${PROJECT_ROOT}/metadata/ace_constrained_reranking_effect_gate.json}"
STAGE="${1:-}"

export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}/packages/pyglmy/src${PYTHONPATH:+:${PYTHONPATH}}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python environment not found: ${PYTHON_BIN}" >&2
  exit 2
fi

CAL_RUN="${PROJECT_ROOT}/${PIPELINE_ROOT}/rerank/topology_constrained_calibration_v1"
CONF_RUN="${PROJECT_ROOT}/${PIPELINE_ROOT}/rerank/topology_constrained_confirmation_v1"
CAL_OUT="${CAL_RUN}/constrained"
CONF_OUT="${CONF_RUN}/constrained"

run_rerank() {
  local config="$1"
  "${PYTHON_BIN}" -m generation.rerank_cli preflight \
    --root "${PROJECT_ROOT}" --config "${config}" --backend ace
  "${PYTHON_BIN}" -m generation.rerank_cli run \
    --root "${PROJECT_ROOT}" --config "${config}" --backend ace --retry-failed
}

run_semantics() {
  local config="$1"
  "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/run_constrained_reranker.py" build-semantics \
    --root "${PROJECT_ROOT}" --run-config "${config}" \
    --selector-config "${SELECTOR_CONFIG}" --device "${CLAP_DEVICE:-cuda:0}"
}

case "${STAGE}" in
  prepare-prompts)
    "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/run_constrained_reranker.py" prepare-prompts \
      --root "${PROJECT_ROOT}" --selector-config "${SELECTOR_CONFIG}"
    ;;
  calibration-run)
    run_rerank "${CALIBRATION_CONFIG}"
    ;;
  calibration-semantics)
    run_semantics "${CALIBRATION_CONFIG}"
    ;;
  calibration-apply)
    "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/run_constrained_reranker.py" apply \
      --root "${PROJECT_ROOT}" --run-config "${CALIBRATION_CONFIG}" \
      --selector-config "${SELECTOR_CONFIG}"
    ;;
  freeze-selector)
    "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/run_constrained_reranker.py" freeze-calibration \
      --root "${PROJECT_ROOT}" --selector-config "${SELECTOR_CONFIG}" \
      --output-dir "${CAL_OUT}" --frozen-selector "${FROZEN_SELECTOR}"
    ;;
  confirmation-run)
    if [[ ! -f "${FROZEN_SELECTOR}" ]]; then
      echo "A supported frozen calibration selector is required." >&2
      exit 2
    fi
    run_rerank "${CONFIRMATION_CONFIG}"
    ;;
  confirmation-semantics)
    run_semantics "${CONFIRMATION_CONFIG}"
    ;;
  confirmation-apply)
    "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/run_constrained_reranker.py" apply \
      --root "${PROJECT_ROOT}" --run-config "${CONFIRMATION_CONFIG}" \
      --selector-config "${SELECTOR_CONFIG}" --frozen-selector "${FROZEN_SELECTOR}"
    ;;
  init-evidence)
    "${PYTHON_BIN}" -m generation.rerank_cli init-evidence \
      --root "${PROJECT_ROOT}" --config "${CONFIRMATION_CONFIG}" --backend ace
    ;;
  quality-template)
    "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/build_reranking_noninferiority_metrics.py" \
      --root "${PROJECT_ROOT}" --run-root "${CONF_RUN}" \
      --prompt-manifest "${PROJECT_ROOT}/${PIPELINE_ROOT}/prompts/confirmation.csv" \
      --model-revision 365dea6ef167def6676140ed93bbc43f84dabb28 \
      --device "${CLAP_DEVICE:-cuda:0}" \
      --output "${CONF_RUN}/noninferiority_quality_template.csv"
    ;;
  final-metrics)
    if [[ -z "${QUALITY_TABLE:-}" ]]; then
      echo "Set QUALITY_TABLE to the completed blinded quality CSV." >&2
      exit 2
    fi
    "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/build_reranking_noninferiority_metrics.py" \
      --root "${PROJECT_ROOT}" --run-root "${CONF_RUN}" \
      --prompt-manifest "${PROJECT_ROOT}/${PIPELINE_ROOT}/prompts/confirmation.csv" \
      --model-revision 365dea6ef167def6676140ed93bbc43f84dabb28 \
      --device "${CLAP_DEVICE:-cuda:0}" --quality-table "${QUALITY_TABLE}" \
      --output "${CONF_RUN}/noninferiority_metrics.csv" \
      --audit-output "${CONF_RUN}/noninferiority_metrics.audit.json"
    ;;
  evaluate-evidence)
    "${PYTHON_BIN}" -m generation.rerank_cli evaluate-evidence \
      --root "${PROJECT_ROOT}" --config "${CONFIRMATION_CONFIG}" --backend ace \
      --noninferiority-table "${CONF_RUN}/noninferiority_metrics.csv"
    ;;
  issue-gate)
    "${PYTHON_BIN}" -m generation.rerank_cli issue-gate \
      --root "${PROJECT_ROOT}" --config "${CONFIRMATION_CONFIG}" --backend ace \
      --selection-contract "${CONF_OUT}/selection_contract.json" \
      --gate-output "${GATE_OUTPUT}"
    ;;
  *)
    echo "Usage: $0 {prepare-prompts|calibration-run|calibration-semantics|calibration-apply|freeze-selector|confirmation-run|confirmation-semantics|confirmation-apply|init-evidence|quality-template|final-metrics|evaluate-evidence|issue-gate}" >&2
    exit 2
    ;;
esac
