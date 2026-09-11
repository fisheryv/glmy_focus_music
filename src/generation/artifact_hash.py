"""Stable hashes for directory-shaped generation artifacts."""

from __future__ import annotations

import hashlib
from pathlib import Path

from .ltsn_contract import LTSNContractError


def sha256_directory(path: Path) -> str:
    """Hash a directory by sorted relative paths and file contents."""

    root = path.resolve()
    files = sorted(item for item in root.rglob("*") if item.is_file())
    if not files:
        raise LTSNContractError(f"artifact directory is empty: {root}")
    digest = hashlib.sha256()
    for item in files:
        relative = item.relative_to(root).as_posix().encode()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()
