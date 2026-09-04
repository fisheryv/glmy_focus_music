#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/ACE-Step-1.5/.venv/bin/python}"
CONFIG="${CONFIG:-configs/ace_rerank_180s.toml}"
RUN_ID="${RUN_ID:-}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python environment not found: ${PYTHON_BIN}" >&2
  echo "Create the ACE-Step environment and install the root project first." >&2
  exit 2
fi

COMMON_ARGS=(
  --root "${PROJECT_ROOT}"
  --config "${CONFIG}"
  --backend ace
)
if [[ -n "${RUN_ID}" ]]; then
  COMMON_ARGS+=(--run-id "${RUN_ID}")
fi

"${PYTHON_BIN}" -m generation.rerank_cli preflight "${COMMON_ARGS[@]}"
"${PYTHON_BIN}" -m generation.rerank_cli plan "${COMMON_ARGS[@]}"
"${PYTHON_BIN}" -m generation.rerank_cli generate "${COMMON_ARGS[@]}" --retry-failed
"${PYTHON_BIN}" -m generation.rerank_cli score "${COMMON_ARGS[@]}"

echo "Exact 18-D scoring finished; no gate was issued."
echo "Run evaluate-evidence, then issue-gate after quality/prompt/diversity metrics are ready."
