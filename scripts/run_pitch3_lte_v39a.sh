#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
exec "${PYTHON_BIN:-python}" "${PROJECT_ROOT}/scripts/run_pitch3_lte_v39a.py" "$@"
