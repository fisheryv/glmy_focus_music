#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export LTE_TRAIN_CONFIG="${LTE_TRAIN_CONFIG:-${PROJECT_ROOT}/configs/pitch3_lte_v36_tcr.toml}"
export LTE_V33_RUN_ROOT="${LTE_V36_TCR_RUN_ROOT:-${PROJECT_ROOT}/runs/pitch3_lte_v36_tcr}"
export LTE_V33_SEEDS="${LTE_V36_TCR_SEEDS:-20260935 20260936 20260937}"
export LTE_VERSION_LABEL="V3.6-TCR"

exec bash "${PROJECT_ROOT}/scripts/run_pitch3_lte_v33.sh" "$@"
