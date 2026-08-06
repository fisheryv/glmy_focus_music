from pathlib import Path

import numpy as np
import pytest

from features.batch import (
    FeatureBatchError,
    _balanced_rows,
    _read_npz,
    _sha256,
    _write_npz_atomic,
    fit_state_model,
)
from features.contracts import FeatureExtractionConfig

pytest.importorskip("sklearn")


def test_balanced_rows_uses_strict_equal_group_counts() -> None:
    values = {
        "a": [np.arange(20, dtype=np.float32).reshape(10, 2)],
        "b": [np.arange(12, dtype=np.float32).reshape(6, 2)],
        "c": [np.arange(16, dtype=np.float32).reshape(8, 2)],
    }

    selected, counts = _balanced_rows(values, maximum_per_group=9, seed=7)

    assert counts == {"a": 6, "b": 6, "c": 6}
    assert selected.shape == (18, 2)


def test_deterministic_npz_has_stable_hash(tmp_path: Path) -> None:
    path = tmp_path / "values.npz"
    arrays = {
        "b": np.array([3, 4], dtype=np.int16),
        "a": np.arange(12, dtype=np.float32).reshape(3, 4),
    }

    _write_npz_atomic(path, arrays)
    first_hash = _sha256(path)
    _write_npz_atomic(path, arrays)

    assert _sha256(path) == first_hash
    with np.load(path, allow_pickle=False) as loaded:
        assert loaded.files == ["a", "b"]
        np.testing.assert_array_equal(loaded["a"], arrays["a"])


def test_corrupt_npz_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "corrupt.npz"
    path.write_bytes(b"not a zip archive")

    with pytest.raises(FeatureBatchError, match="invalid feature archive"):
        _read_npz(path)


def _write_training_features(root: Path, segment_id: str) -> dict[str, str]:
    rng = np.random.default_rng(42)
    acoustic_path = root / f"{segment_id}_acoustic.npz"
    rhythm_path = root / f"{segment_id}_rhythm.npz"
    modulation_path = root / f"{segment_id}_modulation.npz"
    _write_npz_atomic(
        acoustic_path,
        {
            "vectors": rng.normal(size=(100, 12)).astype(np.float32),
            "valid": np.ones(100, dtype=np.bool_),
        },
    )
    rhythm_valid = np.ones((100, 8), dtype=np.bool_)
    rhythm_valid[::7, 4:6] = False
    _write_npz_atomic(
        rhythm_path,
        {
            "vectors": rng.normal(size=(100, 8)).astype(np.float32),
            "valid": rhythm_valid,
        },
    )
    _write_npz_atomic(
        modulation_path,
        {
            "key_band_energies": rng.random((50, 3), dtype=np.float32),
            "valid": np.ones(50, dtype=np.bool_),
        },
    )
    return {
        "acoustic_relative_path": acoustic_path.relative_to(root).as_posix(),
        "rhythm_relative_path": rhythm_path.relative_to(root).as_posix(),
        "modulation_relative_path": modulation_path.relative_to(root).as_posix(),
    }


def test_state_fit_uses_only_discovery_180_and_reuses_model(tmp_path: Path) -> None:
    paths = _write_training_features(tmp_path, "discovery")
    rows = [
        {
            "segment_id": "discovery",
            "track_id": "track_a",
            "group": "focus",
            "split": "discovery",
            "scale_seconds": "180.0",
            "input_sha256": "a" * 64,
            "status": "extracted",
            **paths,
        },
        {
            "segment_id": "validation",
            "track_id": "track_b",
            "group": "focus",
            "split": "validation",
            "scale_seconds": "180.0",
            "input_sha256": "b" * 64,
            "status": "extracted",
            "acoustic_relative_path": "missing-acoustic.npz",
            "rhythm_relative_path": "missing-rhythm.npz",
            "modulation_relative_path": "missing-modulation.npz",
        },
    ]
    config = FeatureExtractionConfig(
        acoustic_pca_components=4,
        acoustic_clusters=4,
        rhythm_clusters=3,
        max_fit_windows_per_group=100,
    )
    model_path = tmp_path / "model.npz"
    metadata_path = tmp_path / "model.json"

    _, first_hash, metadata = fit_state_model(
        rows,
        root=tmp_path,
        config=config,
        model_path=model_path,
        metadata_path=metadata_path,
        overwrite=False,
    )
    _, second_hash, second_metadata = fit_state_model(
        rows,
        root=tmp_path,
        config=config,
        model_path=model_path,
        metadata_path=metadata_path,
        overwrite=False,
    )

    assert metadata["training_segment_ids"] == ["discovery"]
    assert metadata["training_split"] == "discovery"
    assert metadata["training_scale_seconds"] == 180.0
    assert second_hash == first_hash
    assert second_metadata["training_sha256"] == metadata["training_sha256"]


def test_state_fit_rejects_manifest_without_discovery_180(tmp_path: Path) -> None:
    with pytest.raises(FeatureBatchError, match="discovery 180-second"):
        fit_state_model(
            [
                {
                    "segment_id": "validation",
                    "split": "validation",
                    "scale_seconds": "180.0",
                    "status": "extracted",
                }
            ],
            root=tmp_path,
            config=FeatureExtractionConfig(),
            model_path=tmp_path / "model.npz",
            metadata_path=tmp_path / "model.json",
            overwrite=False,
        )
