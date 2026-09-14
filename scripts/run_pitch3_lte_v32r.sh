#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export LTE_V32_RUN_ROOT="${LTE_V32R_RUN_ROOT:-${PROJECT_ROOT}/runs/pitch3_lte_v32r}"
export LTE_TRAIN_CONFIG="${PROJECT_ROOT}/configs/pitch3_lte_v32r.toml"
exec bash "${PROJECT_ROOT}/scripts/run_pitch3_lte_v32.sh" "$@"
