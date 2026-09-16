#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export LTE_TRAIN_CONFIG="${LTE_TRAIN_CONFIG:-${PROJECT_ROOT}/configs/pitch3_lte_v37_dte.toml}"
export LTE_V33_RUN_ROOT="${LTE_V37_DTE_RUN_ROOT:-${PROJECT_ROOT}/runs/pitch3_lte_v37_dte}"
export LTE_V33_SEEDS="${LTE_V37_DTE_SEEDS:-20260938 20260939 20260940}"
export LTE_VERSION_LABEL="V3.7-DTE"

exec bash "${PROJECT_ROOT}/scripts/run_pitch3_lte_v33.sh" "$@"
