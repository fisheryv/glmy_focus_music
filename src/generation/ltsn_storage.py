"""Storage-safe materialization and cleanup helpers for the LTSN pipeline."""

from __future__ import annotations

import errno
import os
import shutil
from pathlib import Path

from .ltsn_contract import LTSNContractError, sha256_file

MATERIALIZE_MODES = ("auto", "reflink", "hardlink", "copy")


def _remove_temporary(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def _reflink(source: Path, target: Path) -> None:
    """Create a Linux copy-on-write clone without silently falling back."""

    if os.name != "posix":
        raise OSError(errno.ENOTSUP, "reflink is only available on supported POSIX filesystems")
    import fcntl

    ficlone = 0x40049409
    with source.open("rb") as source_handle, target.open("xb") as target_handle:
        fcntl.ioctl(target_handle.fileno(), ficlone, source_handle.fileno())
    shutil.copystat(source, target, follow_symlinks=True)


def materialize_file(
    source: Path,
    target: Path,
    *,
    mode: str = "auto",
    expected_sha256: str | None = None,
) -> str:
    """Materialize an immutable input with reflink/hardlink/copy fallback.

    ``auto`` tries a copy-on-write reflink, then a same-filesystem hardlink, and
    finally a physical copy. Every new target is hash-verified before it is
    atomically installed. Existing matching targets are reused; mismatches are
    rejected instead of overwritten.
    """

    if mode not in MATERIALIZE_MODES:
        raise ValueError(f"unsupported materialization mode: {mode}")
    source = source.resolve()
    target = target.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    source_sha256 = expected_sha256 or sha256_file(source)
    if expected_sha256 and sha256_file(source) != expected_sha256:
        raise LTSNContractError(f"source SHA-256 mismatch: {source}")
    if target.exists():
        if not target.is_file() or sha256_file(target) != source_sha256:
            raise LTSNContractError(f"existing materialized file is hash-mismatched: {target}")
        return "existing"

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.materializing")
    _remove_temporary(temporary)
    attempts = ("reflink", "hardlink", "copy") if mode == "auto" else (mode,)
    errors: list[str] = []
    for method in attempts:
        try:
            if method == "reflink":
                _reflink(source, temporary)
            elif method == "hardlink":
                os.link(source, temporary)
            else:
                shutil.copy2(source, temporary)
            if sha256_file(temporary) != source_sha256:
                raise LTSNContractError(
                    f"materialized file failed SHA-256 verification: {temporary}"
                )
            os.replace(temporary, target)
            return method
        except (OSError, LTSNContractError) as exc:
            _remove_temporary(temporary)
            errors.append(f"{method}: {exc}")
            if mode != "auto":
                raise LTSNContractError(
                    f"unable to materialize {source} with {method}: {exc}"
                ) from exc
    raise LTSNContractError(
        f"unable to materialize {source} at {target}; " + "; ".join(errors)
    )


def remove_tree_within(path: Path, allowed_root: Path) -> None:
    """Remove an intermediate tree only when it is strictly below its run root."""

    resolved = path.resolve()
    root = allowed_root.resolve()
    if resolved == root or not resolved.is_relative_to(root):
        raise LTSNContractError(f"refusing cleanup outside the exact-work root: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)


def remove_generated_audio(path: Path, allowed_root: Path) -> None:
    """Remove only an unreferenced generator output inside its declared run root."""

    resolved = path.resolve()
    root = allowed_root.resolve()
    if not resolved.is_relative_to(root):
        raise LTSNContractError(f"refusing to delete generated audio outside {root}: {resolved}")
    resolved.unlink(missing_ok=True)

