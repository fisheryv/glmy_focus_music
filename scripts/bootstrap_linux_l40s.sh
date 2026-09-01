#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"
ACE_REPOSITORY="https://github.com/fisheryv/ACE-Step-1.5.git"
ACE_REVISION="a5632cda3084f1088e69b2057dde7047e1bb4839"
PYGLMY_REPOSITORY="https://github.com/fisheryv/pyglmy.git"
PYGLMY_REVISION="49bd5ea7617906f09940dcc9b9718bbfc1482d6f"

command -v git >/dev/null || { echo "git is required" >&2; exit 2; }
command -v "${PYTHON_BIN}" >/dev/null || { echo "${PYTHON_BIN} is required" >&2; exit 2; }

mkdir -p "${PROJECT_ROOT}/packages"

clone_at_revision() {
  local repository="$1"
  local revision="$2"
  local destination="$3"
  if [[ ! -d "${destination}/.git" ]]; then
    git clone "${repository}" "${destination}"
  fi
  if [[ "$(git -C "${destination}" remote get-url origin)" != "${repository}" ]]; then
    echo "Unexpected origin for ${destination}" >&2
    exit 2
  fi
  if [[ "$(git -C "${destination}" rev-parse HEAD)" != "${revision}" ]]; then
    if [[ -n "$(git -C "${destination}" status --porcelain)" ]]; then
      echo "Refusing to change a dirty checkout: ${destination}" >&2
      exit 2
    fi
    git -C "${destination}" fetch origin "${revision}"
    git -C "${destination}" checkout --detach "${revision}"
  fi
}

clone_at_revision "${PYGLMY_REPOSITORY}" "${PYGLMY_REVISION}" "${PROJECT_ROOT}/packages/pyglmy"
clone_at_revision "${ACE_REPOSITORY}" "${ACE_REVISION}" "${PROJECT_ROOT}/ACE-Step-1.5"

PATCH="${PROJECT_ROOT}/patches/ace-step-1.5-topology-corrector.patch"
if git -C "${PROJECT_ROOT}/ACE-Step-1.5" apply --reverse --check "${PATCH}" >/dev/null 2>&1; then
  echo "ACE-Step topology patch is already applied"
elif git -C "${PROJECT_ROOT}/ACE-Step-1.5" apply --check "${PATCH}"; then
  git -C "${PROJECT_ROOT}/ACE-Step-1.5" apply "${PATCH}"
else
  echo "ACE-Step checkout is neither clean-patchable nor already patched" >&2
  exit 2
fi

"${PYTHON_BIN}" -m pip install --user "uv==0.8.13"
UV_BIN="${UV_BIN:-$("${PYTHON_BIN}" -c 'import shutil; print(shutil.which("uv") or "")')}"
if [[ -z "${UV_BIN}" ]]; then
  echo "uv was installed but is not visible; set UV_BIN explicitly" >&2
  exit 2
fi

"${UV_BIN}" sync --project "${PROJECT_ROOT}/ACE-Step-1.5" --locked
ACE_PYTHON="${PROJECT_ROOT}/ACE-Step-1.5/.venv/bin/python"
"${UV_BIN}" pip install --python "${ACE_PYTHON}" \
  -e "${PROJECT_ROOT}/packages/pyglmy[tda]" \
  -e "${PROJECT_ROOT}[audio,stats,tda,topology-guidance,repro,dev]"

"${ACE_PYTHON}" "${PROJECT_ROOT}/scripts/verify_linux_l40s.py" \
  --root "${PROJECT_ROOT}" \
  --allow-missing-data

cat <<EOF
Environment created successfully.
Activate it with:
  source "${PROJECT_ROOT}/ACE-Step-1.5/.venv/bin/activate"
Then download and verify the dataset:
  python scripts/prepare_release_dataset.py
EOF
