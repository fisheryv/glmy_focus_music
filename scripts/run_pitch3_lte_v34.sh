#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export LTE_TRAIN_CONFIG="${LTE_TRAIN_CONFIG:-${PROJECT_ROOT}/configs/pitch3_lte_v34.toml}"
export LTE_V33_RUN_ROOT="${LTE_V34_RUN_ROOT:-${PROJECT_ROOT}/runs/pitch3_lte_v34}"
export LTE_V33_SEEDS="${LTE_V34_SEEDS:-20260926 20260927 20260928}"
export LTE_VERSION_LABEL="V3.4"

exec bash "${PROJECT_ROOT}/scripts/run_pitch3_lte_v33.sh" "$@"
