#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export LTE_TRAIN_CONFIG="${LTE_TRAIN_CONFIG:-${PROJECT_ROOT}/configs/pitch3_lte_v35r.toml}"
export LTE_V33_RUN_ROOT="${LTE_V35R_RUN_ROOT:-${PROJECT_ROOT}/runs/pitch3_lte_v35r}"
export LTE_V33_SEEDS="${LTE_V35R_SEEDS:-20260932 20260933 20260934}"
export LTE_VERSION_LABEL="V3.5R"

exec bash "${PROJECT_ROOT}/scripts/run_pitch3_lte_v33.sh" "$@"
