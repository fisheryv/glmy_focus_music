#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export LTE_TRAIN_CONFIG="${LTE_TRAIN_CONFIG:-${PROJECT_ROOT}/configs/pitch3_lte_v35.toml}"
export LTE_V33_RUN_ROOT="${LTE_V35_RUN_ROOT:-${PROJECT_ROOT}/runs/pitch3_lte_v35}"
export LTE_V33_SEEDS="${LTE_V35_SEEDS:-20260929 20260930 20260931}"
export LTE_VERSION_LABEL="V3.5"

exec bash "${PROJECT_ROOT}/scripts/run_pitch3_lte_v33.sh" "$@"
