from __future__ import annotations

from pathlib import Path

from scripts.verify_linux_l40s import verify


def test_missing_scorer_release_is_reported_without_traceback(tmp_path: Path) -> None:
    reproducibility = tmp_path / "reproducibility"
    reproducibility.mkdir()
    (reproducibility / "release_manifest.toml").write_text(
        """
[sources.pyglmy]
revision = "missing"

[sources.ace_step]
revision = "missing"

[sources.dataset]
sha256s_sha256 = "missing"

[scorer]
release_manifest = "metadata/focus_path_homology_fingerprint_v2_release.json"
profile_sha256 = "missing"
""".strip()
        + "\n",
        encoding="utf-8",
    )

    payload = verify(tmp_path, allow_missing_data=True)

    checks = {item["name"]: item for item in payload["checks"]}
    assert payload["status"] == "failed"
    assert checks["scorer_release_manifest"]["observed"] == "missing"
    assert checks["scorer_release_manifest"]["ok"] is False
    assert checks["scorer_profile"]["observed"] == "missing"
    assert checks["scorer_profile"]["ok"] is False
    assert "scorer_dimensions" not in checks
